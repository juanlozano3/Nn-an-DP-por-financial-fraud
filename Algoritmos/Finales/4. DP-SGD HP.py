import os
import time
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
from itertools import product
from tqdm.auto import tqdm  # auto: usa barra bonita en Colab/Jupyter o terminal

# -----------------------------------------------------------------------------
# Configuración de recursos de hardware
# -----------------------------------------------------------------------------
NUM_AVAILABLE_CPUS = os.cpu_count() or 1
DEFAULT_CPU_TARGET = int(os.environ.get("DP_HP_NUM_CPUS", NUM_AVAILABLE_CPUS))
DEFAULT_INTER_THREADS = int(
    os.environ.get("DP_HP_INTER_THREADS", max(1, DEFAULT_CPU_TARGET // 2))
)
DEFAULT_INTRA_THREADS = int(os.environ.get("DP_HP_INTRA_THREADS", DEFAULT_CPU_TARGET))

DEFAULT_INTER_THREADS = max(1, DEFAULT_INTER_THREADS)
DEFAULT_INTRA_THREADS = max(1, DEFAULT_INTRA_THREADS)

os.environ.setdefault("OMP_NUM_THREADS", str(DEFAULT_INTRA_THREADS))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(DEFAULT_INTRA_THREADS))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(DEFAULT_INTER_THREADS))

try:
    tf.config.threading.set_inter_op_parallelism_threads(DEFAULT_INTER_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(DEFAULT_INTRA_THREADS)
    tf.config.set_soft_device_placement(True)
except RuntimeError:
    pass

AUTOTUNE = (
    tf.data.AUTOTUNE if hasattr(tf.data, "AUTOTUNE") else tf.data.experimental.AUTOTUNE
)


# -----------------------------------------------------------------------------
# Utilidades de diagnóstico
# -----------------------------------------------------------------------------
def _maybe_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def hardware_summary():
    psutil = _maybe_import_psutil()
    cpu_freq = None
    cpu_percent = None
    ram_total = None
    ram_available = None

    if psutil:
        try:
            cpu_freq = psutil.cpu_freq()
        except Exception:
            cpu_freq = None
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_percent = None
        try:
            virtual_mem = psutil.virtual_memory()
            ram_total = getattr(virtual_mem, "total", None)
            ram_available = getattr(virtual_mem, "available", None)
        except Exception:
            ram_total = None
            ram_available = None

    return {
        "logical_cpus": NUM_AVAILABLE_CPUS,
        "tf_intra_threads": DEFAULT_INTRA_THREADS,
        "tf_inter_threads": DEFAULT_INTER_THREADS,
        "cpu_freq_current_mhz": cpu_freq.current if cpu_freq else None,
        "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else None,
        "cpu_percent": cpu_percent,
        "ram_total_bytes": ram_total,
        "ram_available_bytes": ram_available,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "tf_env_intra": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "tf_env_inter": os.environ.get("TF_NUM_INTEROP_THREADS"),
    }


def print_hardware_summary(summary_dict):
    print("\n" + "=" * 80)
    print("RESUMEN DE RECURSOS PARA LA BÚSQUEDA DE HIPERPARÁMETROS")
    print("=" * 80)
    print(f"  • CPUs lógicas detectadas: {summary_dict['logical_cpus']}")
    print(
        f"  • Threads TF intra-op / inter-op: "
        f"{summary_dict['tf_intra_threads']} / {summary_dict['tf_inter_threads']}"
    )
    print(
        f"  • TF_NUM_INTRAOP_THREADS / TF_NUM_INTEROP_THREADS: "
        f"{summary_dict['tf_env_intra']} / {summary_dict['tf_env_inter']}"
    )
    print(f"  • OMP_NUM_THREADS: {summary_dict['omp_num_threads']}")

    if summary_dict.get("cpu_freq_current_mhz") is not None:
        print(
            f"  • Frecuencia CPU actual/max (MHz): "
            f"{summary_dict['cpu_freq_current_mhz']:.0f} / {summary_dict['cpu_freq_max_mhz']:.0f}"
        )
    if summary_dict.get("cpu_percent") is not None:
        print(f"  • Uso CPU global al inicio: {summary_dict['cpu_percent']:.1f}%")
    if summary_dict.get("ram_total_bytes") is not None:
        gb = 1024**3
        print(
            f"  • RAM total / disponible (GB): "
            f"{summary_dict['ram_total_bytes'] / gb:.2f} / "
            f"{summary_dict['ram_available_bytes'] / gb:.2f}"
        )
    print("=" * 80 + "\n")


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
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.cache()
        dataset = dataset.prefetch(AUTOTUNE)
        options = tf.data.Options()
        try:
            threading_opts = options.threading
        except AttributeError:
            threading_opts = getattr(options, "experimental_threading", None)
        if threading_opts is not None:
            try:
                threading_opts.private_threadpool_size = DEFAULT_INTRA_THREADS
            except AttributeError:
                pass
            try:
                threading_opts.max_intra_op_parallelism = DEFAULT_INTRA_THREADS
            except AttributeError:
                pass
        else:
            warnings.warn(
                "No se pudo ajustar tf.data.Options.threading; se usará configuración por defecto.",
                RuntimeWarning,
            )
        try:
            options.threading = threading_opts
        except AttributeError:
            pass
        dataset = dataset.with_options(options)
        return dataset

    return input_fn


def main():
    """
    Main function to run DP-SGD training with tf.Estimator.
    """
    resource_summary = hardware_summary()
    print_hardware_summary(resource_summary)

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
    batch_size = 1024
    steps_per_epoch = X_train.shape[0] // batch_size
    # ============================================
    # HPO grid: l2_norm_clip, noise_multiplier, learning_rate, threshold, epochs
    # ============================================
    CLIP_GRID = [0.5, 1.0, 1.5]
    NOISE_GRID = [0.8, 1.0, 1.1]
    LR_GRID = [0.01, 0.1, 0.25]
    THRESHOLD_GRID = [0.2]
    EPOCHS_GRID = [15, 25, 30]
    grid_combos = list(
        product(CLIP_GRID, NOISE_GRID, LR_GRID, THRESHOLD_GRID, EPOCHS_GRID)
    )
    total_trials = len(grid_combos)

    best = {
        "score": -float("inf"),
        "params": None,
        "epochs": None,
        "val_loss": float("inf"),
        "summary": "",
    }

    with mlflow.start_run(run_name="DP-SGD-HPO"):
        # Log contexto de dataset
        mlflow.log_param("n_train", int(X_train.shape[0]))
        mlflow.log_param("n_test", int(X_test.shape[0]))
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("selection_metric", "val_loss")
        mlflow.log_param("epsilon_threshold", 5.0)
        mlflow.log_param("cpu_available", DEFAULT_CPU_TARGET)
        mlflow.log_param("tf_inter_threads", DEFAULT_INTER_THREADS)
        mlflow.log_param("tf_intra_threads", DEFAULT_INTRA_THREADS)
        for k, v in resource_summary.items():
            if (
                k not in {"logical_cpus", "tf_intra_threads", "tf_inter_threads"}
                and v is not None
            ):
                mlflow.log_param(f"resource_{k}", v)

        # Prepara input fns (reutilizables)
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

        session_config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=DEFAULT_INTER_THREADS,
            intra_op_parallelism_threads=DEFAULT_INTRA_THREADS,
            allow_soft_placement=True,
            device_count={"CPU": DEFAULT_CPU_TARGET},
        )
        run_config = tf_estimator.RunConfig(
            session_config=session_config,
            log_step_count_steps=steps_per_epoch,
        )

    trial_num = 0
    print("\n" + "=" * 80)
    print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
    print(f"Total de combinaciones a probar: {total_trials}")
    print("=" * 80 + "\n")

    pbar = tqdm(
        grid_combos,
        desc="HPO Progress",
        total=total_trials,
        unit="trial",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
    )

    for clip, noise, lr, threshold, total_epochs in pbar:
        trial_num += 1
        pbar.set_description(
            f"Trial {trial_num}/{total_trials} | clip={clip} noise={noise} lr={lr} thr={threshold} ep={total_epochs}"
        )
        if trial_num == 1:
            print(
                f">> Ejecutando búsqueda con hilos TF intra/inter: "
                f"{DEFAULT_INTRA_THREADS}/{DEFAULT_INTER_THREADS} y batch_size={batch_size}"
            )

        trial_params = {
            "l2_norm_clip": clip,
            "noise_multiplier": noise,
            "num_microbatches": 32,
            "learning_rate": lr,
            "unroll_microbatches": False,
            "pred_threshold": threshold,
        }

        run_name = (
            f"clip={clip}_noise={noise}_lr={lr}_thr={threshold}_ep={total_epochs}"
        )
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log params del trial
            mlflow.log_param("l2_norm_clip", clip)
            mlflow.log_param("noise_multiplier", noise)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("threshold", threshold)
            mlflow.log_param("num_microbatches", 32)
            mlflow.log_param("epochs", total_epochs)

            fraud_classifier = tf_estimator.Estimator(
                model_fn=model, params=trial_params, config=run_config
            )

            for epoch in range(1, total_epochs + 1):
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
                    mlflow.log_metric(
                        "recall_class_0", report["0"]["recall"], step=epoch
                    )
                    mlflow.log_metric("f1_class_0", report["0"]["f1-score"], step=epoch)
                    mlflow.log_metric(
                        "precision_class_1", report["1"]["precision"], step=epoch
                    )
                    mlflow.log_metric(
                        "recall_class_1", report["1"]["recall"], step=epoch
                    )
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
                    epsilon = compute_epsilon(
                        epoch,
                        noise,
                        X_train.shape[0],
                        batch_size,
                        delta=1.0 / X_train.shape[0],
                    )
                    mlflow.log_metric("epsilon", epsilon, step=epoch)
                    if epsilon > 5.0:
                        mlflow.log_param("early_stopped", True)
                        mlflow.log_param("early_stop_epoch", epoch)
                        mlflow.log_param("early_stop_reason", "epsilon > 5.0")
                        break
                else:
                    epsilon = 0.0

            preds = list(fraud_classifier.predict(input_fn=eval_input))
            y_pred = [1 if p["prob"][0] > threshold else 0 for p in preds]
            report = classification_report(
                y_test[: len(y_pred)], y_pred, output_dict=True
            )

            mlflow.log_metric("precision_class_0_final", report["0"]["precision"])
            mlflow.log_metric("recall_class_0_final", report["0"]["recall"])
            mlflow.log_metric("f1_class_0_final", report["0"]["f1-score"])
            mlflow.log_metric("precision_class_1_final", report["1"]["precision"])
            mlflow.log_metric("recall_class_1_final", report["1"]["recall"])
            mlflow.log_metric("f1_class_1_final", report["1"]["f1-score"])
            mlflow.log_metric("precision_macro_final", report["macro avg"]["precision"])
            mlflow.log_metric("recall_macro_final", report["macro avg"]["recall"])
            mlflow.log_metric("f1_macro_final", report["macro avg"]["f1-score"])

            cm = confusion_matrix(y_test[: len(y_pred)], y_pred)
            cm_df = pd.DataFrame(
                cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
            )
            cm_csv_path = f"cm_clip{clip}_noise{noise}_lr{lr}_thr{threshold}_ep{total_epochs}_final.csv"
            cm_df.to_csv(cm_csv_path)
            mlflow.log_artifact(cm_csv_path)
            os.remove(cm_csv_path)

            trial_score = -float(eval_results["loss"])
            mlflow.log_metric("trial_score", -trial_score)
            mlflow.log_metric("val_loss_final", float(eval_results["loss"]))

            final_auprc = float(eval_results["auprc"])
            final_auroc = float(eval_results["auroc"])
            final_acc = float(eval_results["accuracy"])
            final_loss = float(eval_results["loss"])
            final_epsilon = epsilon if noise > 0 else 0.0

            is_best = trial_score > best["score"]
            if is_best:
                best["score"] = trial_score
                best["params"] = trial_params
                best["epochs"] = epoch
                best["val_loss"] = float(eval_results["loss"])
                best["summary"] = (
                    f"Loss={final_loss:.4f}, AUPRC={final_auprc:.4f}, "
                    f"AUROC={final_auroc:.4f}, ACC={final_acc:.4f}, EPS={final_epsilon:.4f}"
                )

            status_msg = f"Loss={final_loss:.4f} | AUPRC={final_auprc:.4f}"
            if is_best:
                status_msg += " ⭐ BEST"
            pbar.set_postfix_str(status_msg)

            if (
                is_best
                or trial_num == 1
                or trial_num % 10 == 0
                or trial_num == total_trials
            ):
                print("\n" + "=" * 80, flush=True)
                print(f"TRIAL {trial_num}/{total_trials} COMPLETADO", flush=True)
                print("=" * 80, flush=True)
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
                print(
                    f"\n{'*** NUEVO MEJOR MODELO ***' if is_best else 'Comparación con el mejor:'}",
                    flush=True,
                )
                if is_best:
                    print(
                        f"  Val Loss mejorado:   {final_loss:.4f} (nuevo mejor)",
                        flush=True,
                    )
                else:
                    print(f"  Val Loss actual:      {final_loss:.4f}", flush=True)
                    print(
                        f"  Val Loss mejor:       {best.get('val_loss', float('inf')):.4f}",
                        flush=True,
                    )
                print("=" * 80 + "\n", flush=True)

    if not best["params"]:
        raise RuntimeError("No se encontró un mejor conjunto de hiperparámetros.")

    with mlflow.start_run(run_name="FINAL_best", nested=True):
        for k, v in best["params"].items():
            mlflow.log_param(k, v)
        mlflow.log_param("epochs", best["epochs"])

        fraud_classifier = tf_estimator.Estimator(
            model_fn=model, params=best["params"], config=run_config
        )
        final_noise = best["params"]["noise_multiplier"]
        for epoch in range(1, best["epochs"] + 1):
            fraud_classifier.train(input_fn=train_input, steps=steps_per_epoch)
            eval_results = fraud_classifier.evaluate(
                input_fn=eval_input, steps=eval_steps
            )
            mlflow.log_metric("eval_loss", float(eval_results["loss"]), step=epoch)
            mlflow.log_metric("accuracy", float(eval_results["accuracy"]), step=epoch)
            mlflow.log_metric("auroc", float(eval_results["auroc"]), step=epoch)
            mlflow.log_metric("auprc", float(eval_results["auprc"]), step=epoch)

            if final_noise > 0:
                epsilon = compute_epsilon(
                    epoch,
                    noise,
                    X_train.shape[0],
                    batch_size,
                    delta=1.0 / X_train.shape[0],
                )

                mlflow.log_metric("epsilon", epsilon, step=epoch)
                if epsilon > 5.0:
                    print(
                        f"\n>> Early stopping in final training: epsilon ({epsilon:.4f}) > 5.0 at epoch {epoch}"
                    )
                    break

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
        best["final_confusion_matrix"] = cm_df

    print("\n" + "=" * 80)
    print("MEJOR MODELO ENCONTRADO (DP-SGD)")
    print("=" * 80)
    print(f"Resumen métricas: {best['summary']}")
    print("Hiperparámetros seleccionados:")
    for k, v in best["params"].items():
        print(f"  • {k}: {v}")
    print(f"  • Epochs entrenados: {best['epochs']}")
    if best.get("final_confusion_matrix") is not None:
        print("\nMatriz de confusión (mejor modelo):")
        print(best["final_confusion_matrix"])
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
