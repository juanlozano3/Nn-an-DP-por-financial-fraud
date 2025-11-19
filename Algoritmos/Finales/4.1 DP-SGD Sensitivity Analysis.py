import os
import time
import tempfile
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib

import mlflow
import mlflow.tensorflow

import sys, os

GDP_REPO = "../../Deep-Learning-with-GDP-Tensorflow"
sys.path.append(GDP_REPO)
from gdp_accountant import compute_epsP, compute_epsilon


def define_model(features):
    """Define the model architecture."""
    n_features = features["x"].shape[-1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(224, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="out"),
        ]
    )
    return model


def model(features, labels, mode, params):
    """
    Estimator model_fn.
    Builds the network, defines per-example loss for DP-SGD, sets the DP optimizer,
    and returns the proper EstimatorSpec for TRAIN/EVAL/PREDICT.
    """
    # --- Build model ---
    model = define_model(features)

    # Forward pass -> probabilities in [0,1] (final layer is sigmoid)
    probs = model(features["x"])

    # Prediction threshold (can be overridden from outside; default 0.5 or your tuned value)
    thr = float(params.get("pred_threshold", 0.5))
    class_ids = tf.cast(probs > thr, tf.int64)

    predictions = {
        "class_ids": class_ids,
        "prob": probs,
    }

    # --- MODO PREDICT ---
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # --- Prepare labels for loss/metrics ---
    labels = tf.cast(labels, tf.float32)  # ensure float for BCE
    labels = tf.reshape(labels, (-1, 1))  # shape: (batch, 1)

    # --- Loss Function ---
    # We use reduction=NONE to get a vector of losses; DP-SGD needs per-example losses.
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    vector_loss = loss_fn(y_true=labels, y_pred=probs)  # shape: (batch,)
    scalar_loss = tf.reduce_mean(vector_loss)  # reported loss

    # --- Metrics (use consistent dtypes/shapes) ---
    labels_int = tf.cast(tf.reshape(labels, [-1]), tf.int64)  # (batch,)
    preds_int = tf.reshape(class_ids, [-1])  # (batch,)
    acc = tf.compat.v1.metrics.accuracy(labels=labels_int, predictions=preds_int)

    # AUROC/AUPRC on probabilities; metrics expect 1-D same-length tensors
    auroc = tf.compat.v1.metrics.auc(
        labels=tf.reshape(labels, [-1]),
        predictions=tf.reshape(probs, [-1]),
        curve="ROC",
        name="auroc",
    )
    auprc = tf.compat.v1.metrics.auc(
        labels=tf.reshape(labels, [-1]),
        predictions=tf.reshape(probs, [-1]),
        curve="PR",
        name="auprc",
        summation_method="careful_interpolation",  # Fix PR-AUC warning
    )

    # --- OPTIMIZADOR DP ---
    optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=params["l2_norm_clip"],
        noise_multiplier=params["noise_multiplier"],
        num_microbatches=params["num_microbatches"],
        learning_rate=params["learning_rate"],
        unroll_microbatches=params["unroll_microbatches"],
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss=vector_loss, global_step=global_step)

    # --- TRAIN ---
    if mode == tf_estimator.ModeKeys.TRAIN:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op
        )

    # --- EVAL ---
    elif mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(
            mode=mode,
            loss=scalar_loss,
            eval_metric_ops={"accuracy": acc, "auroc": auroc, "auprc": auprc},
        )


def make_input_fn(X, y, batch_size, shuffle=True, repeat=True):
    """
    Create input function for the Estimator.
    """

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"x": X}, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def train_and_evaluate_model(
    X_train, y_train, X_test, y_test, params, batch_size, total_epochs, prob_threshold, model_dir=None
):
    """
    Entrena y evalúa un modelo con los parámetros dados.
    Retorna: epsilon final, AUPRC final, AUROC final
    
    Args:
        model_dir: Directorio único para cada modelo (asegura estado limpio)
    """
    # Crear directorio único para este modelo si no se proporciona
    if model_dir is None:
        model_dir = tempfile.mkdtemp()
    
    # Crear nuevo estimator para cada experimento con directorio único (limpia el estado)
    fraud_classifier = tf_estimator.Estimator(
        model_fn=model, 
        params=params,
        model_dir=model_dir
    )

    train_input = make_input_fn(X_train, y_train, batch_size, shuffle=True, repeat=True)
    eval_input = make_input_fn(X_test, y_test, batch_size, shuffle=False, repeat=False)
    eval_steps = max(1, X_test.shape[0] // batch_size)
    steps_per_epoch = X_train.shape[0] // batch_size

    # Training loop
    final_auprc = None
    final_auroc = None
    final_epsilon = None

    for epoch in range(1, total_epochs + 1):
        # Train
        fraud_classifier.train(input_fn=train_input, steps=steps_per_epoch)

        # Evaluate
        eval_results = fraud_classifier.evaluate(input_fn=eval_input, steps=eval_steps)

        # Guardar métricas del último epoch
        if epoch == total_epochs:
            final_auprc = float(eval_results.get("auprc", 0.0))
            final_auroc = float(eval_results.get("auroc", 0.0))

        # Calcular epsilon
        if params["noise_multiplier"] > 0:
            final_epsilon = compute_epsilon(
                epoch,
                params["noise_multiplier"],
                X_train.shape[0],
                batch_size,
                delta=1.0 / X_train.shape[0],
            )

    # Limpiar directorio temporal
    try:
        shutil.rmtree(model_dir)
    except:
        pass

    return final_epsilon, final_auprc, final_auroc


def sensitivity_analysis():
    """
    Realiza análisis de sensibilidad variando un hiperparámetro a la vez.
    """
    mlflow.set_experiment("DP-SGD-Sensitivity-Analysis")

    # Load and prepare dataset (igual que en el código original)
    data = pd.read_csv("../../Datos/2/Base.csv")
    X = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # One hot for categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical_cols])
    # transform numeric variables
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    # Join of both datas
    X_num = X[numeric_cols].values
    X = np.hstack([X_num, X_cat])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    # Balanceo (igual que en el código original)
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)

    X_majority = X_train_df[y_train_series == 0]
    X_minority = X_train_df[y_train_series == 1]

    y_majority = y_train_series[y_train_series == 0]
    y_minority = y_train_series[y_train_series == 1]

    # Downsample of the class
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority,
        y_majority,
        replace=False,
        n_samples=len(y_minority),
        random_state=42,
    )

    X_train_balanced = pd.concat([X_majority_downsampled, X_minority])
    y_train_balanced = pd.concat([y_majority_downsampled, y_minority])

    # Shuffle after concatenation
    shuf_idx = np.random.RandomState(42).permutation(len(y_train_balanced))
    X_train = X_train_balanced.to_numpy().astype(np.float32)[shuf_idx]
    y_train = y_train_balanced.to_numpy().astype(np.int32)[shuf_idx]

    print("=" * 80)
    print("ANÁLISIS DE SENSIBILIDAD DE HIPERPARÁMETROS - DP-SGD")
    print("=" * 80)
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    print(f"Tamaño del conjunto de prueba: {len(X_test)}")
    print("=" * 80)

    # ============================================================
    # PARÁMETROS BASE (fijos para todos los experimentos)
    # ============================================================
    BASE_PARAMS = {
        "l2_norm_clip": 1.2,
        "noise_multiplier": 1.2,
        "num_microbatches": 32,
        "learning_rate": 0.25,
        "unroll_microbatches": False,
    }
    BASE_BATCH_SIZE = 256
    BASE_EPOCHS = 5
    BASE_THRESHOLD = 0.2

    # ============================================================
    # DEFINIR VALORES A PROBAR PARA CADA HIPERPARÁMETRO
    # ============================================================
    # Cada hiperparámetro tendrá 3 valores: bajo, medio, alto
    HYPERPARAMETER_VALUES = {
        "l2_norm_clip": [0.8, 1.2, 1.6],  # Clipping norm
        "noise_multiplier": [0.8, 1.2, 1.6],  # Noise multiplier
        "learning_rate": [0.1, 0.25, 0.5],  # Learning rate
        "epochs": [3, 5, 7],  # Number of epochs
        "pred_threshold": [0.1, 0.2, 0.3],  # Prediction threshold
    }

    # ============================================================
    # ALMACENAR RESULTADOS
    # ============================================================
    results = []

    # ============================================================
    # ANÁLISIS PARA CADA HIPERPARÁMETRO
    # ============================================================
    for hyperparam_name, hyperparam_values in HYPERPARAMETER_VALUES.items():
        print(f"\n{'=' * 80}")
        print(f"Analizando: {hyperparam_name}")
        print(f"Valores a probar: {hyperparam_values}")
        print(f"{'=' * 80}")

        for value in hyperparam_values:
            print(f"\n  Probando {hyperparam_name} = {value}...")

            # Crear copia de parámetros base
            params = BASE_PARAMS.copy()
            batch_size = BASE_BATCH_SIZE
            total_epochs = BASE_EPOCHS
            prob_threshold = BASE_THRESHOLD

            # Ajustar el hiperparámetro que estamos variando
            if hyperparam_name == "l2_norm_clip":
                params["l2_norm_clip"] = value
            elif hyperparam_name == "noise_multiplier":
                params["noise_multiplier"] = value
            elif hyperparam_name == "learning_rate":
                params["learning_rate"] = value
            elif hyperparam_name == "epochs":
                total_epochs = value
            elif hyperparam_name == "pred_threshold":
                prob_threshold = value
                params["pred_threshold"] = value

            # Entrenar y evaluar
            epsilon, auprc, auroc = train_and_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                params,
                batch_size,
                total_epochs,
                prob_threshold,
            )

            # Guardar resultado
            result = {
                "hyperparameter": hyperparam_name,
                "value": value,
                "epsilon": epsilon,
                "auprc": auprc,
                "auroc": auroc,
            }
            results.append(result)

            print(f"    ε = {epsilon:.4f}, AUPRC = {auprc:.4f}, AUROC = {auroc:.4f}")

    # ============================================================
    # CONVERTIR RESULTADOS A DATAFRAME
    # ============================================================
    results_df = pd.DataFrame(results)

    # Guardar resultados a CSV
    results_df.to_csv("sensitivity_analysis_results.csv", index=False)
    print(f"\n{'=' * 80}")
    print("Resultados guardados en: sensitivity_analysis_results.csv")
    print(f"{'=' * 80}")

    # ============================================================
    # GENERAR GRÁFICAS
    # ============================================================
    print("\nGenerando gráficas...")

    # Configurar estilo
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Crear directorio para gráficas
    os.makedirs("sensitivity_plots", exist_ok=True)

    # Mapeo de nombres de hiperparámetros a nombres descriptivos
    hyperparam_display_names = {
        "l2_norm_clip": "L2_Norm_Clipping",
        "noise_multiplier": "Noise_Multiplier",
        "learning_rate": "Learning_Rate",
        "epochs": "Number_of_Epochs",
        "pred_threshold": "Prediction_Threshold",
    }

    # Para cada hiperparámetro, crear gráficas: epsilon, AUPRC y AUROC
    for hyperparam_name in HYPERPARAMETER_VALUES.keys():
        hyperparam_data = results_df[results_df["hyperparameter"] == hyperparam_name]
        display_name = hyperparam_display_names.get(hyperparam_name, hyperparam_name)
        readable_name = hyperparam_name.replace("_", " ").title()

        # Gráfica 1: Epsilon vs Hiperparámetro
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hyperparam_data["value"],
            hyperparam_data["epsilon"],
            marker="o",
            linewidth=2.5,
            markersize=12,
            color="red",
            label="ε (Epsilon)",
        )
        ax.set_xlabel(f"{readable_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("ε (Epsilon - Privacy Budget)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Impacto de {readable_name} en Privacidad (ε)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        filename = f"sensitivity_plots/01_HP_{display_name}_vs_Epsilon.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardada: {filename}")

        # Gráfica 2: AUPRC vs Hiperparámetro
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hyperparam_data["value"],
            hyperparam_data["auprc"],
            marker="s",
            linewidth=2.5,
            markersize=12,
            color="green",
            label="AUPRC",
        )
        ax.set_xlabel(f"{readable_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUPRC (Area Under PR Curve)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Impacto de {readable_name} en Rendimiento (AUPRC)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        filename = f"sensitivity_plots/02_HP_{display_name}_vs_AUPRC.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardada: {filename}")

        # Gráfica 3: AUROC vs Hiperparámetro
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            hyperparam_data["value"],
            hyperparam_data["auroc"],
            marker="^",
            linewidth=2.5,
            markersize=12,
            color="purple",
            label="AUROC",
        )
        ax.set_xlabel(f"{readable_name}", fontsize=12, fontweight="bold")
        ax.set_ylabel("AUROC (Area Under ROC Curve)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Impacto de {readable_name} en Rendimiento (AUROC)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        filename = f"sensitivity_plots/03_HP_{display_name}_vs_AUROC.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardada: {filename}")

        print(f"  ✓ Gráficas generadas para {readable_name}")

    # ============================================================
    # GRÁFICA COMPARATIVA: Todas las métricas juntas
    # ============================================================
    print("\nGenerando gráficas comparativas...")

    for hyperparam_name in HYPERPARAMETER_VALUES.keys():
        hyperparam_data = results_df[results_df["hyperparameter"] == hyperparam_name]
        display_name = hyperparam_display_names.get(hyperparam_name, hyperparam_name)
        readable_name = hyperparam_name.replace("_", " ").title()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Epsilon
        ax1.plot(
            hyperparam_data["value"],
            hyperparam_data["epsilon"],
            marker="o",
            linewidth=2.5,
            markersize=12,
            color="red",
            label="ε (Epsilon)",
        )
        ax1.set_xlabel(f"{readable_name}", fontsize=12, fontweight="bold")
        ax1.set_ylabel("ε (Epsilon - Privacy Budget)", fontsize=12, fontweight="bold")
        ax1.set_title("Impacto en Privacidad", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)

        # AUPRC
        ax2.plot(
            hyperparam_data["value"],
            hyperparam_data["auprc"],
            marker="s",
            linewidth=2.5,
            markersize=12,
            color="green",
            label="AUPRC",
        )
        ax2.set_xlabel(f"{readable_name}", fontsize=12, fontweight="bold")
        ax2.set_ylabel("AUPRC (Area Under PR Curve)", fontsize=12, fontweight="bold")
        ax2.set_title("Impacto en Rendimiento", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.suptitle(
            f"Análisis de Sensibilidad: {readable_name}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        filename = f"sensitivity_plots/04_HP_{display_name}_Comparison_Epsilon_vs_AUPRC.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Guardada: {filename}")

    print(f"\n{'=' * 80}")
    print("✓ Todas las gráficas guardadas en: sensitivity_plots/")
    print(f"{'=' * 80}")

    # ============================================================
    # LOGGING A MLFLOW
    # ============================================================
    with mlflow.start_run(run_name="Sensitivity_Analysis_Complete"):
        # Log parámetros base
        for key, value in BASE_PARAMS.items():
            mlflow.log_param(f"base_{key}", value)
        mlflow.log_param("base_batch_size", BASE_BATCH_SIZE)
        mlflow.log_param("base_epochs", BASE_EPOCHS)
        mlflow.log_param("base_threshold", BASE_THRESHOLD)

        # Log resultados como artifact
        mlflow.log_artifact("sensitivity_analysis_results.csv")

        # Log gráficas como artifacts
        hyperparam_display_names = {
            "l2_norm_clip": "L2_Norm_Clipping",
            "noise_multiplier": "Noise_Multiplier",
            "learning_rate": "Learning_Rate",
            "epochs": "Number_of_Epochs",
            "pred_threshold": "Prediction_Threshold",
        }
        
        for hyperparam_name in HYPERPARAMETER_VALUES.keys():
            display_name = hyperparam_display_names.get(hyperparam_name, hyperparam_name)
            mlflow.log_artifact(f"sensitivity_plots/01_HP_{display_name}_vs_Epsilon.png")
            mlflow.log_artifact(f"sensitivity_plots/02_HP_{display_name}_vs_AUPRC.png")
            mlflow.log_artifact(f"sensitivity_plots/03_HP_{display_name}_vs_AUROC.png")
            mlflow.log_artifact(f"sensitivity_plots/04_HP_{display_name}_Comparison_Epsilon_vs_AUPRC.png")

        print("\n✓ Resultados y gráficas logueados en MLflow")

    print(f"\n{'=' * 80}")
    print("ANÁLISIS DE SENSIBILIDAD COMPLETADO")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    sensitivity_analysis()

