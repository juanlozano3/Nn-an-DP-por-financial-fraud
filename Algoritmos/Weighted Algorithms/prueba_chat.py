"""
Imports originales (limpiados) + nuevos
"""

import os
import time
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from tensorflow_privacy.privacy.optimizers import dp_optimizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_class_weight  # <- NUEVO

import mlflow
import mlflow.tensorflow

# GDP accountant (ruta de tu repo)
GDP_REPO = "../../Deep-Learning-with-GDP-Tensorflow"
sys.path.append(GDP_REPO)
from gdp_accountant import compute_epsP, compute_epsilon


def define_model(features):
    """
    Define the model architecture.
    """
    # Nota: asegura tupla con coma al final en input_shape
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(features["x"].shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def model(features, labels, mode, params):
    """
    Model Function for tf.Estimator.
    Define el modelo, la función de pérdida, el optimizador y las salidas según el modo.
    """
    # Construir el modelo
    model_keras = define_model(features)

    # Forward pass: logits = salida activada sigmoide
    logits = model_keras(features["x"])

    # Definir predicciones (para modo PREDICT)
    predictions = {
        "class_ids": tf.cast(logits > 0.2, tf.int64),  # umbral configurable
        "logits": logits,
    }

    # --- MODO PREDICT ---
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # --- PREP LABELS ---
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))

    # --- FUNCIÓN DE PÉRDIDA (por ejemplo) ---
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    vector_loss = loss_fn(y_true=labels, y_pred=logits)  # shape [batch, 1]

    # --- SAMPLE WEIGHTS desde features (si no hay, usa 1.0) ---
    w = features.get("w", None)
    if w is None:
        w = tf.ones_like(vector_loss, dtype=tf.float32)
    else:
        w = tf.cast(w, tf.float32)
        w = tf.reshape(w, (-1, 1))

    # Normaliza para que E[w] ≈ 1 (mantiene escala estable para LR/DP)
    w = w / (tf.reduce_mean(w) + 1e-8)

    # Aplica pesos por ejemplo (clave para DP)
    weighted_vector_loss = vector_loss * w

    # Pérdida escalar (para logs / EstimatorSpec)
    scalar_loss = tf.reduce_mean(weighted_vector_loss)

    # --- MÉTRICAS ---
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=tf.cast(labels, tf.int64),
        predictions=predictions["class_ids"],
    )

    # --- OPTIMIZADOR DP ---
    optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=params["l2_norm_clip"],
        noise_multiplier=params["noise_multiplier"],
        num_microbatches=params["num_microbatches"],
        learning_rate=params["learning_rate"],
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    # Minimiza la pérdida POR-EJEMPLO ya ponderada
    train_op = optimizer.minimize(loss=weighted_vector_loss, global_step=global_step)

    # --- MODO TRAIN ---
    if mode == tf_estimator.ModeKeys.TRAIN:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op
        )

    # --- MODO EVAL ---
    elif mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, eval_metric_ops={"accuracy": accuracy}
        )


def make_input_fn(X, y, batch_size, shuffle=True, repeat=True, sample_weight=None):
    """
    Create input function for the Estimator (opcionalmente con sample weights).
    """

    def input_fn():
        feats = {"x": X}
        if sample_weight is not None:
            feats["w"] = sample_weight  # añadimos pesos por ejemplo

        dataset = tf.data.Dataset.from_tensor_slices((feats, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def main():
    """
    Main function to run DP-SGD training with tf.Estimator.
    """
    mlflow.set_experiment("DP-Fraud-Detection")

    # Load and prepare dataset
    data = pd.read_csv("../../Datos/2/Base.csv")

    # Split features and labels
    X = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # One-hot categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical_cols])

    # Escala numéricas
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Combina
    X_num = X[numeric_cols].values
    X = np.hstack([X_num, X_cat])

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convertir a numpy arrays y asegurarse de los tipos
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)

    # Distribución original (ANTES de downsampling)
    y_train_orig = y_train.copy()

    unique_pre, counts_pre = np.unique(y_train, return_counts=True)
    print("Antes del balanceo:", dict(zip(unique_pre, counts_pre)))

    # =========================
    # Balanceo de clases (Downsample)
    # =========================

    # 1. Convertir a DataFrame/Series para usar pd.concat
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)

    # 2. Dividir por clases
    X_majority = X_train_df[y_train_series == 0]
    X_minority = X_train_df[y_train_series == 1]

    y_majority = y_train_series[y_train_series == 0]
    y_minority = y_train_series[y_train_series == 1]

    # 3. Downsample de la clase mayoritaria
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority,
        y_majority,
        replace=False,  # Sin reemplazo
        n_samples=len(y_minority),  # Igualar al número de la clase minoritaria
        random_state=42,  # Reproducibilidad
    )

    # 4. Concatenar para obtener el set balanceado
    X_train_balanced = pd.concat([X_majority_downsampled, X_minority])
    y_train_balanced = pd.concat([y_majority_downsampled, y_minority])

    # 5. Convertir de nuevo a numpy arrays y sobrescribir las variables originales
    X_train = X_train_balanced.to_numpy().astype(np.float32)
    y_train = y_train_balanced.to_numpy().astype(np.int32)

    # 6. Imprimir las nuevas formas
    print("X train: ", len(X_train))
    print("X Test: ", len(X_test))
    print("y Train: ", len(y_train))
    print("y Test: ", len(y_test))

    # Después del balanceo
    unique_post, counts_post = np.unique(y_train, return_counts=True)
    print("Después del balanceo:", dict(zip(unique_post, counts_post)))

    # =========================
    # Pesos de clase + sample weights (post-downsample) con mezcla
    # =========================
    classes = np.array([0, 1])

    # Pesos "balanced" con la distribución original
    w_orig = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train_orig
    )
    w0_orig, w1_orig = float(w_orig[0]), float(w_orig[1])

    # Pesos "balanced" con la distribución post-downsample
    w_post = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    w0_post, w1_post = float(w_post[0]), float(w_post[1])

    # Mezcla (0 = solo post; 1 = solo original)
    alpha = 0.5
    w0 = (1 - alpha) * w0_post + alpha * w0_orig
    w1 = (1 - alpha) * w1_post + alpha * w1_orig

    # Cap de razón para evitar sobrecompensación
    max_ratio = 3.0
    ratio = max(w1 / (w0 + 1e-8), w0 / (w1 + 1e-8))
    if ratio > max_ratio:
        if w1 >= w0:
            w1 = w0 * max_ratio
        else:
            w0 = w1 * max_ratio

    # Vector de sample weights para el set YA downsampleado
    sample_weight_train = np.where(y_train == 0, w0, w1).astype(np.float32)

    # Normaliza para que E[w] ≈ 1 (estabilidad para LR/DP)
    sample_weight_train /= sample_weight_train.mean() + 1e-8

    print(f"[Weights] w0={w0:.3f}  w1={w1:.3f}  mean={sample_weight_train.mean():.3f}")

    # =========================
    # Entrenamiento
    # =========================
    batch_size = 256
    total_epochs = 5
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)

    # DP-SGD parameters
    params = {
        "l2_norm_clip": 1.2,
        "noise_multiplier": 1.1,
        "num_microbatches": 32,
        "learning_rate": 0.15,
    }

    with mlflow.start_run(run_name="DP-SGD"):
        # Log params
        mlflow.log_param("l2_norm_clip", params["l2_norm_clip"])
        mlflow.log_param("noise_multiplier", params["noise_multiplier"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("num_microbatches", params["num_microbatches"])
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", total_epochs)
        mlflow.log_param("alg", "SGD-PCA-VarSelection+Downsample+Weights")

        # Estimator
        fraud_classifier = tf_estimator.Estimator(model_fn=model, params=params)

        # Bucle de entrenamiento
        for epoch in range(1, total_epochs + 1):
            start_time = time.time()

            # Train (con sample weights)
            fraud_classifier.train(
                input_fn=make_input_fn(
                    X_train,
                    y_train,
                    batch_size,
                    shuffle=True,
                    repeat=True,
                    sample_weight=sample_weight_train,
                ),
                steps=steps_per_epoch,
            )

            end_time = time.time()
            print(f"Epoch {epoch}/{total_epochs} - Time: {end_time - start_time:.2f}s")

            # Evaluate (sin pesos, métricas estándar)
            eval_results = fraud_classifier.evaluate(
                input_fn=make_input_fn(
                    X_test, y_test, batch_size, shuffle=False, repeat=False
                ),
                steps=max(1, X_test.shape[0] // batch_size),
            )
            print(f"Evaluation: {eval_results}")

            # Classification report
            predictions = list(
                fraud_classifier.predict(
                    input_fn=make_input_fn(
                        X_test, y_test, batch_size, shuffle=False, repeat=False
                    )
                )
            )
            mlflow.log_metric(
                "accuracy", float(eval_results.get("accuracy", 0.0)), step=epoch
            )

            y_pred = [1 if p["logits"][0] > 0.1 else 0 for p in predictions]

            report = classification_report(
                y_test[: len(y_pred)], y_pred, output_dict=True
            )

            print("\nClassification Report:")
            print(classification_report(y_test[: len(y_pred)], y_pred, digits=4))

            cm = confusion_matrix(y_test[: len(y_pred)], y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            # Convertir la matriz a DataFrame para que se vea bien
            cm_df = pd.DataFrame(
                cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
            )

            # --- Pérdida ---
            mlflow.log_metric(
                "eval_loss", float(eval_results.get("loss", 0.0)), step=epoch
            )

            # --- Accuracy (ya la tienes) ---
            mlflow.log_metric(
                "accuracy", float(eval_results.get("accuracy", 0.0)), step=epoch
            )

            # --- Precisión, recall y f1 de cada clase ---
            mlflow.log_metric(
                "precision_class_0", float(report["0"]["precision"]), step=epoch
            )
            mlflow.log_metric(
                "recall_class_0", float(report["0"]["recall"]), step=epoch
            )
            mlflow.log_metric("f1_class_0", float(report["0"]["f1-score"]), step=epoch)

            mlflow.log_metric(
                "precision_class_1", float(report["1"]["precision"]), step=epoch
            )
            mlflow.log_metric(
                "recall_class_1", float(report["1"]["recall"]), step=epoch
            )
            mlflow.log_metric("f1_class_1", float(report["1"]["f1-score"]), step=epoch)

            # --- Métricas agregadas (macro / weighted) ---
            mlflow.log_metric(
                "precision_macro", float(report["macro avg"]["precision"]), step=epoch
            )
            mlflow.log_metric(
                "recall_macro", float(report["macro avg"]["recall"]), step=epoch
            )
            mlflow.log_metric(
                "f1_macro", float(report["macro avg"]["f1-score"]), step=epoch
            )

            mlflow.log_metric(
                "precision_weighted",
                float(report["weighted avg"]["precision"]),
                step=epoch,
            )
            mlflow.log_metric(
                "recall_weighted", float(report["weighted avg"]["recall"]), step=epoch
            )
            mlflow.log_metric(
                "f1_weighted", float(report["weighted avg"]["f1-score"]), step=epoch
            )

            # Guardar como CSV y subir a MLflow
            cm_csv_path = f"confusion_matrix_epoch_{epoch}.csv"
            cm_df.to_csv(cm_csv_path)
            mlflow.log_artifact(cm_csv_path)
            os.remove(cm_csv_path)

            # --- Privacidad (ε) ---
            if params["noise_multiplier"] > 0:
                epsilon = compute_epsilon(
                    epoch,
                    params["noise_multiplier"],
                    X_train.shape[0],
                    batch_size,
                    1e-6,
                )
                mlflow.log_metric("epsilon", float(epsilon), step=epoch)
                print(f"DP-SGD Privacy after {epoch} epochs: ε = {epsilon:.2f}")


if __name__ == "__main__":
    main()
