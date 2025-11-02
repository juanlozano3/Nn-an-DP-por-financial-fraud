import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc  # Para liberación explícita de memoria

# Suprimir warnings de TensorFlow y otras librerías
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import estimator as tf_estimator

# Configuración de GPU para mejor rendimiento
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento de memoria GPU para evitar asignar toda la memoria al inicio
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Configurar uso de GPU visible (usar la primera GPU disponible)
        if len(gpus) > 0:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"✓ GPU detectada y configurada: {gpus[0].name}")
            print(f"  Memoria GPU configurada para crecimiento dinámico")
    except RuntimeError as e:
        # Si hay error en configuración de GPU, continúa con CPU
        print(f"Advertencia: Error configurando GPU: {e}")
        print("  Continuando con CPU...")
        gpus = []
else:
    print("⚠ No se detectó GPU. Usando CPU.")

# Estrategia 1: Configurar threads de TensorFlow para aprovechar todos los cores (si no hay GPU)
if not gpus:
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    print(f"TensorFlow configurado para usar {num_cores} cores CPU")
else:
    # Con GPU, usar menos threads para no competir con la GPU
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.dp_query import gaussian_query

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
from itertools import product
from tqdm.auto import tqdm  # auto: usa barra bonita en Colab/Jupyter o terminal

# Configurar tqdm para que sea menos verboso
tqdm.pandas = lambda *args, **kwargs: None


GDP_REPO = "../../Deep-Learning-with-GDP-Tensorflow"
sys.path.append(GDP_REPO)
from gdp_accountant import compute_epsP, compute_epsilon


def define_model(features):
    """Define the model architecture."""
    n_features = features["x"].shape[-1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(288, activation="relu", name="dense_1"),
            tf.keras.layers.Dropout(0.3, name="dropout_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.Dropout(0.2, name="dropout_2"),
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
        summation_method="careful_interpolation",
        name="auroc",
    )
    auprc = tf.compat.v1.metrics.auc(
        labels=tf.reshape(labels, [-1]),
        predictions=tf.reshape(probs, [-1]),
        curve="PR",
        summation_method="careful_interpolation",
        name="auprc",
    )

    # --- OPTIMIZADOR DP ---
    # Calculate stddev for Gaussian noise: stddev = l2_norm_clip * noise_multiplier
    stddev = params["l2_norm_clip"] * params["noise_multiplier"]
    dp_sum_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=params["l2_norm_clip"],
        stddev=stddev
    )
    
    # Note: When using dp_sum_query, l2_norm_clip and noise_multiplier are already included
    # in the query, so they should not be passed separately to the optimizer
    optimizer = dp_optimizer.DPGradientDescentOptimizer(
        dp_sum_query,
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
    Optimizado con prefetch para mejor rendimiento.
    """
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(({"x": X}, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(X), 10000))  # Buffer más pequeño para mejor rendimiento
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        # Estrategia 2: Optimizar tf.data pipeline con prefetch
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # Prepara batches en paralelo
        return dataset

    return input_fn


def main():
    """
    Main function to run DP-SGD training with tf.Estimator.
    """
    mlflow.set_experiment("DP-Fraud-Detection-Final")

    # Load and prepare dataset
    data = pd.read_csv("../../Datos/2/Base.csv")
    # Split features and labels
    X = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # One hot for categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical_cols])
    # teansform numeric variables
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
    unique_pre, counts_pre = np.unique(y_train, return_counts=True)
    print("Antes del balanceo:", dict(zip(unique_pre, counts_pre)))

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

    # Shuffle after concatenation to avoid ordered blocks
    shuf_idx = np.random.RandomState(42).permutation(len(y_train_balanced))
    X_train = X_train_balanced.to_numpy().astype(np.float32)[shuf_idx]
    y_train = y_train_balanced.to_numpy().astype(np.int32)[shuf_idx]

    # Shapes & balance after resampling
    print("\nPreparación de datos completada:")
    print(f"  • X train: {len(X_train)}")
    print(f"  • X Test:  {len(X_test)}")
    print(f"  • y Train: {len(y_train)}")
    print(f"  • y Test:  {len(y_test)}")
    # Después del balanceo
    unique_post, counts_post = np.unique(y_train, return_counts=True)
    print(f"  • Balanceo: {dict(zip(unique_post, counts_post))}\n")

    # Training params
    batch_size = 256
    steps_per_epoch = X_train.shape[0] // batch_size
    # ============================================
    # HPO grid: l2_norm_clip, noise_multiplier, learning_rate, threshold, epochs
    # ============================================
    CLIP_GRID = [0.5, 1.5]
    NOISE_GRID = [ 0.8, 1.1]
    LR_GRID = [0.01, 0.1]
    THRESHOLD_GRID = [0.2]
    EPOCHS_GRID = [15, 25, 30]  # Rango de épocas entre 10 y 30
    grid_combos = list(product(CLIP_GRID, NOISE_GRID, LR_GRID, THRESHOLD_GRID, EPOCHS_GRID))
    total_trials = len(grid_combos)
    print(f"Total number of hyperparameter combinations: {total_trials}")
    print(f"Grid sizes: Clip={len(CLIP_GRID)}, Noise={len(NOISE_GRID)}, LR={len(LR_GRID)}, "
          f"Threshold={len(THRESHOLD_GRID)}, Epochs={len(EPOCHS_GRID)}")

    best = {"score": -float("inf"), "params": None, "epochs": None, "val_loss": float("inf")}

    with mlflow.start_run(run_name="DP-HPO"):
        # Log contexto de dataset
        mlflow.log_param("n_train", int(X_train.shape[0]))
        mlflow.log_param("n_test", int(X_test.shape[0]))
        mlflow.log_param("batch_size", 256)  # lo usas abajo
        mlflow.log_param("selection_metric", "val_loss")  # Métrica usada para selección del mejor modelo
        mlflow.log_param("epsilon_threshold", 5.0)  # Threshold para early stopping

        # Prepara input fns (reutilizables)
        batch_size = 256
        steps_per_epoch = max(1, X_train.shape[0] // batch_size)

        def make_train_input():
            return make_input_fn(
                X_train, y_train, batch_size, shuffle=True, repeat=True
            )

        def make_eval_input():
            return make_input_fn(
                X_test, y_test, batch_size, shuffle=False, repeat=False
            )

        train_input = make_train_input()
        eval_input = make_eval_input()
        eval_steps = max(1, X_test.shape[0] // batch_size)

    trial_num = 0
    print("\n" + "="*80)
    print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
    print(f"Total de combinaciones a probar: {total_trials}")
    print("="*80 + "\n")
    
    # Crear barra de progreso con información detallada
    pbar = tqdm(
        grid_combos,
        desc="HPO Progress",
        total=total_trials,
        unit="trial",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=100
    )
    
    for clip, noise, lr, threshold, total_epochs in pbar:
        trial_num += 1
        
        # Actualizar descripción de la barra con información del trial actual
        pbar.set_description(
            f"Trial {trial_num}/{total_trials} | clip={clip} noise={noise} lr={lr} thr={threshold} ep={total_epochs}"
        )
        trial_params = {
            "l2_norm_clip": clip,
            "noise_multiplier": noise,
            "num_microbatches": 32,
            "learning_rate": lr,
            "unroll_microbatches": False,
            "pred_threshold": threshold,
        }

        run_name = f"clip={clip}_noise={noise}_lr={lr}_thr={threshold}_ep={total_epochs}"
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log params del trial
            mlflow.log_param("l2_norm_clip", clip)
            mlflow.log_param("noise_multiplier", noise)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("num_microbatches", 32)
            mlflow.log_param("epochs", total_epochs)

            fraud_classifier = tf_estimator.Estimator(
                model_fn=model, params=trial_params
            )

            # Estrategia 4: Acumular métricas para batch logging
            metrics_batch = []
            
            # === entrenamiento por épocas ===
            for epoch in range(1, total_epochs + 1):
                t0 = time.time()
                fraud_classifier.train(input_fn=train_input, steps=steps_per_epoch)
                dur = time.time() - t0

                # Eval
                eval_results = fraud_classifier.evaluate(
                    input_fn=eval_input, steps=eval_steps
                )

                # Estrategia 4: Acumular métricas básicas en lugar de loguear inmediatamente
                metrics_batch.append({
                    "eval_loss": float(eval_results["loss"]),
                    "accuracy": float(eval_results["accuracy"]),
                    "auroc": float(eval_results["auroc"]),
                    "auprc": float(eval_results["auprc"]),
                    "step": epoch
                })
                
                # Loguear en batch cada 5 épocas o en la última
                if epoch % 5 == 0 or epoch == total_epochs:
                    for metric in metrics_batch:
                        mlflow.log_metric("eval_loss", metric["eval_loss"], step=metric["step"])
                        mlflow.log_metric("accuracy", metric["accuracy"], step=metric["step"])
                        mlflow.log_metric("auroc", metric["auroc"], step=metric["step"])
                        mlflow.log_metric("auprc", metric["auprc"], step=metric["step"])
                    metrics_batch = []  # Limpiar batch después de loguear

                # Métricas detalladas solo cada 5 épocas o en la última (optimización 3)
                log_detailed = (epoch % 5 == 0) or (epoch == total_epochs)
                
                if log_detailed:
                    preds = list(fraud_classifier.predict(input_fn=eval_input))
                    y_pred = [1 if p["prob"][0] > threshold else 0 for p in preds]
                    report = classification_report(
                        y_test[: len(y_pred)], y_pred, output_dict=True
                    )

                    mlflow.log_metric(
                        "precision_class_0", report["0"]["precision"], step=epoch
                    )
                    mlflow.log_metric("recall_class_0", report["0"]["recall"], step=epoch)
                    mlflow.log_metric("f1_class_0", report["0"]["f1-score"], step=epoch)
                    mlflow.log_metric(
                        "precision_class_1", report["1"]["precision"], step=epoch
                    )
                    mlflow.log_metric("recall_class_1", report["1"]["recall"], step=epoch)
                    mlflow.log_metric("f1_class_1", report["1"]["f1-score"], step=epoch)
                    mlflow.log_metric(
                        "precision_macro", report["macro avg"]["precision"], step=epoch
                    )
                    mlflow.log_metric(
                        "recall_macro", report["macro avg"]["recall"], step=epoch
                    )
                    mlflow.log_metric(
                        "f1_macro", report["macro avg"]["f1-score"], step=epoch
                    )

                if noise > 0:
                    epsilon = compute_epsP(
                        epoch, noise, X_train.shape[0], batch_size, 1e-6
                    )
                    mlflow.log_metric("epsilon", epsilon, step=epoch)
                    
                    # Early stopping si epsilon > 5
                    if epsilon > 5.0:
                        mlflow.log_param("early_stopped", True)
                        mlflow.log_param("early_stop_epoch", epoch)
                        mlflow.log_param("early_stop_reason", "epsilon > 5.0")
                        break
                else:
                    epsilon = 0.0
            
            # Optimización 1 y 2: Predict, classification_report y confusion matrix solo al final del trial
            preds = list(fraud_classifier.predict(input_fn=eval_input))
            y_pred = [1 if p["prob"][0] > threshold else 0 for p in preds]
            report = classification_report(
                y_test[: len(y_pred)], y_pred, output_dict=True
            )
            
            # Log métricas finales detalladas
            mlflow.log_metric(
                "precision_class_0_final", report["0"]["precision"]
            )
            mlflow.log_metric("recall_class_0_final", report["0"]["recall"])
            mlflow.log_metric("f1_class_0_final", report["0"]["f1-score"])
            mlflow.log_metric(
                "precision_class_1_final", report["1"]["precision"]
            )
            mlflow.log_metric("recall_class_1_final", report["1"]["recall"])
            mlflow.log_metric("f1_class_1_final", report["1"]["f1-score"])
            mlflow.log_metric(
                "precision_macro_final", report["macro avg"]["precision"]
            )
            mlflow.log_metric(
                "recall_macro_final", report["macro avg"]["recall"]
            )
            mlflow.log_metric(
                "f1_macro_final", report["macro avg"]["f1-score"]
            )
            
            # Confusion matrix solo al final (optimización 2)
            cm = confusion_matrix(y_test[: len(y_pred)], y_pred)
            cm_df = pd.DataFrame(
                cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
            )
            cm_csv_path = f"cm_clip{clip}_noise{noise}_lr{lr}_thr{threshold}_ep{total_epochs}_final.csv"
            cm_df.to_csv(cm_csv_path)
            mlflow.log_artifact(cm_csv_path)
            os.remove(cm_csv_path)

            # criterio de selección del trial: usar val_loss (minimizar pérdida)
            trial_score = -float(eval_results["loss"])  # Negativo porque buscamos mínimo pérdida
            mlflow.log_metric("trial_score", -trial_score)  # Log como pérdida positiva
            mlflow.log_metric("val_loss_final", float(eval_results["loss"]))
            
            # Obtener métricas finales del trial
            final_auprc = float(eval_results["auprc"])
            final_auroc = float(eval_results["auroc"])
            final_acc = float(eval_results["accuracy"])
            final_loss = float(eval_results["loss"])
            final_epsilon = epsilon if noise > 0 else 0.0
            
            # Determinar si es el mejor
            is_best = trial_score > best["score"]
            if is_best:  # Mayor score = menor pérdida (porque es negativo)
                best["score"] = trial_score
                best["params"] = trial_params
                best["epochs"] = epoch  # Usar epoch actual en caso de early stopping
                best["val_loss"] = float(eval_results["loss"])
            
            # Actualizar la barra de progreso con información del trial
            status_msg = f"Loss={final_loss:.4f} | AUPRC={final_auprc:.4f}"
            if is_best:
                status_msg += " ⭐ BEST"
            pbar.set_postfix_str(status_msg)
            
            # Imprimir resultados del trial de forma clara y detallada (resumen)
            # Solo imprimir cada N trials o cuando hay un nuevo mejor modelo para no saturar
            if is_best or trial_num == 1 or trial_num % 10 == 0 or trial_num == total_trials:
                print("\n" + "="*80, flush=True)
                print(f"TRIAL {trial_num}/{total_trials} COMPLETADO", flush=True)
                print("="*80, flush=True)
                print(f"Hiperparámetros:", flush=True)
                print(f"  • L2 Norm Clip:     {clip}", flush=True)
                print(f"  • Noise Multiplier:  {noise}", flush=True)
                print(f"  • Learning Rate:     {lr}", flush=True)
                print(f"  • Threshold:         {threshold}", flush=True)
                print(f"  • Epochs:            {epoch}/{total_epochs}", flush=True)
                print(f"\nMétricas Finales del Trial:", flush=True)
                print(f"  • Val Loss:          {final_loss:.4f}", flush=True)
                print(f"  • AUPRC:             {final_auprc:.4f}", flush=True)
                print(f"  • AUROC:             {final_auroc:.4f}", flush=True)
                print(f"  • Accuracy:          {final_acc:.4f}", flush=True)
                if noise > 0:
                    print(f"  • Epsilon (ε):       {final_epsilon:.4f}", flush=True)
                print(f"\n{'*** NUEVO MEJOR MODELO ***' if is_best else 'Comparación con el mejor:'}", flush=True)
                if is_best:
                    print(f"  Val Loss mejorado:   {final_loss:.4f} (nuevo mejor)", flush=True)
                else:
                    print(f"  Val Loss actual:      {final_loss:.4f}", flush=True)
                    print(f"  Val Loss mejor:       {best.get('val_loss', float('inf')):.4f}", flush=True)
                print("="*80 + "\n", flush=True)
            
            # Estrategia 3: Liberar memoria explícitamente después de cada trial
            tf.keras.backend.clear_session()  # Limpia la sesión de TensorFlow
            del fraud_classifier  # Elimina el estimador
            gc.collect()  # Fuerza garbage collection

        # ============================================
        # Re-entrenar final con los mejores HPs encontrados
        # ============================================
        with mlflow.start_run(run_name="FINAL_best", nested=True):
            for k, v in best["params"].items():
                mlflow.log_param(k, v)
            mlflow.log_param("epochs", best["epochs"])

            fraud_classifier = tf_estimator.Estimator(
                model_fn=model, params=best["params"]
            )
            final_noise = best["params"]["noise_multiplier"]
            for epoch in range(1, best["epochs"] + 1):
                fraud_classifier.train(input_fn=train_input, steps=steps_per_epoch)
                eval_results = fraud_classifier.evaluate(
                    input_fn=eval_input, steps=eval_steps
                )
                mlflow.log_metric("eval_loss", float(eval_results["loss"]), step=epoch)
                mlflow.log_metric(
                    "accuracy", float(eval_results["accuracy"]), step=epoch
                )
                mlflow.log_metric("auroc", float(eval_results["auroc"]), step=epoch)
                mlflow.log_metric("auprc", float(eval_results["auprc"]), step=epoch)
                
                # Early stopping también en entrenamiento final si epsilon > 5
                if final_noise > 0:
                    epsilon = compute_epsP(
                        epoch, final_noise, X_train.shape[0], batch_size, 1e-6
                    )
                    mlflow.log_metric("epsilon", epsilon, step=epoch)
                    if epsilon > 5.0:
                        print(f"\n>> Early stopping in final training: epsilon ({epsilon:.4f}) > 5.0 at epoch {epoch}")
                        break

            # Reporte final y CM
            final_threshold = best["params"]["pred_threshold"]
            preds = list(fraud_classifier.predict(input_fn=eval_input))
            y_pred = [1 if p["prob"][0] > final_threshold else 0 for p in preds]
            report_text = classification_report(y_test[: len(y_pred)], y_pred, digits=4)
            print("\nClassification Report (FINAL):\n", report_text)
            mlflow.log_text(report_text, "final_classification_report.txt")

            cm = confusion_matrix(y_test[: len(y_pred)], y_pred)
            cm_df = pd.DataFrame(
                cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
            )
            cm_csv_path = "final_confusion_matrix.csv"
            cm_df.to_csv(cm_csv_path)
            mlflow.log_artifact(cm_csv_path)
            os.remove(cm_csv_path)


if __name__ == "__main__":
    main()
