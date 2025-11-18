import os
import random
import numpy as np
import pandas as pd
from sklearn.utils import resample
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_recall_fscore_support,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
from tqdm.auto import tqdm
import warnings
import logging

# Suprimir warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# =========================
# Configuración de recursos de hardware
# =========================
NUM_AVAILABLE_CPUS = os.cpu_count() or 1
DEFAULT_CPU_TARGET = int(os.environ.get("NUM_CPUS", NUM_AVAILABLE_CPUS))
DEFAULT_INTER_THREADS = int(
    os.environ.get("INTER_THREADS", max(1, DEFAULT_CPU_TARGET // 2))
)
DEFAULT_INTRA_THREADS = int(os.environ.get("INTRA_THREADS", DEFAULT_CPU_TARGET))

# Limitar valores mínimos
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

# =========================
# Semillas (reproducibilidad)
# =========================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Carga y preparación de datos
# =========================
data = pd.read_csv("../../Datos/2/Base.csv")

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"].astype(np.int32)

# Convert the categorical columns to dummies
X = pd.get_dummies(X, drop_first=True)
# Guardar nombres de características para selección de variables
feature_names = X.columns.tolist()

# =========================
# CRÍTICO: Split PRIMERO (evita data leakage)
# =========================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y  # Aumentado a 0.2 (más estándar)
)

# Escalado con ajuste SOLO en train
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
# Reset de índices para alinear
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# =========================
# Balanceo por downsampling (solo en TRAIN)
# =========================
X_train_df = pd.DataFrame(X_train)
y_train_series = pd.Series(y_train)

# 2. Dividir por clases
X_majority = X_train_df[y_train_series == 0]
X_minority = X_train_df[y_train_series == 1]

y_majority = y_train_series[y_train_series == 0]
y_minority = y_train_series[y_train_series == 1]

# Downsample de la clase mayoritaria
X_majority_downsampled, y_majority_downsampled = resample(
    X_majority,
    y_majority,
    replace=False,  # Sin reemplazo
    n_samples=len(y_minority),  # Igualar al número de la clase minoritaria
    random_state=SEED,  # Reproducibilidad
)

# Concatenar para obtener el set balanceado
X_train_balanced = pd.concat([X_majority_downsampled, X_minority])
y_train_balanced = pd.concat([y_majority_downsampled, y_minority])

# Convertir de nuevo a numpy arrays y barajar
shuf_idx = np.random.RandomState(SEED).permutation(len(y_train_balanced))
X_train = X_train_balanced.to_numpy().astype(np.float32)[shuf_idx]
y_train = y_train_balanced.to_numpy().astype(np.int32)[shuf_idx]

# Asegurar tipos en test
X_test = X_test.astype(np.float32)
y_test = y_test.to_numpy().astype(np.int32)

# =========================
# SELECCIÓN DE VARIABLES: VarianceThreshold + Mutual Information
# =========================
print("\n" + "=" * 80)
print("SELECCIÓN DE VARIABLES")
print("=" * 80)
print(f"Características iniciales: {len(feature_names)}")

# Parámetros de selección (configurables vía variables de entorno)
variance_threshold = float(os.environ.get("VARSEL_VARIANCE_THRESHOLD", 0.0))
max_features = int(os.environ.get("VARSEL_MAX_FEATURES", 200))
min_features = int(os.environ.get("VARSEL_MIN_FEATURES", 20))

# Paso 1: VarianceThreshold (eliminar características con baja varianza)
var_selector = VarianceThreshold(threshold=variance_threshold)
X_train = var_selector.fit_transform(X_train)
X_test = var_selector.transform(X_test)
feature_names_after_var = np.array(feature_names)[
    var_selector.get_support(indices=True)
]

if X_train.shape[1] == 0:
    raise RuntimeError(
        "VarianceThreshold eliminó todas las características; reduce el umbral."
    )

print(
    f"Después de VarianceThreshold (threshold={variance_threshold}): {len(feature_names_after_var)} características"
)

# Paso 2: Mutual Information (seleccionar las más informativas)
mi_scores = mutual_info_classif(
    X_train, y_train, discrete_features=False, random_state=SEED
)
mi_scores = np.nan_to_num(mi_scores, nan=0.0)
sorted_indices = np.argsort(mi_scores)[::-1]
positive = np.sum(mi_scores > 0)
candidate_top = len(sorted_indices)
top_k = max(min_features, min(max_features, candidate_top))
if positive > 0:
    top_k = max(min_features, min(max_features, positive))
top_indices = sorted_indices[:top_k]

X_train = X_train[:, top_indices].astype(np.float32)
X_test = X_test[:, top_indices].astype(np.float32)
selected_feature_names = feature_names_after_var[top_indices]
selected_mi_scores = mi_scores[top_indices]

print(
    f"Después de Mutual Information: {len(selected_feature_names)} características seleccionadas"
)
print(f"  • Rango: {min_features} - {max_features} características")
print(f"  • Características con MI > 0: {positive}")

# Mostrar top 10 características seleccionadas
top_10_indices = np.argsort(selected_mi_scores)[::-1][:10]
print("\nTop 10 características seleccionadas (por Mutual Information):")
for idx in top_10_indices:
    print(f"  • {selected_feature_names[idx]}: {selected_mi_scores[idx]:.6f}")

print("=" * 80)

# Imprimir información final
print("\nX train:", X_train.shape, "| X test:", X_test.shape)
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))


# =========================
# Función para resumen de hardware
# =========================
def _maybe_import_psutil():
    try:
        import psutil

        return psutil
    except Exception:
        return None


def hardware_summary():
    """Devuelve información relevante de CPU/RAM para registro y diagnóstico."""
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
    """Imprime en consola la configuración de hardware utilizada."""
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
    print("=" * 80)


# =========================
# Parámetros fijos de arquitectura (encontrados en búsqueda previa)
# =========================
UNITS1 = 288  # Capa 1
UNITS2 = 64  # Capa 2
DROPOUT1 = 0.3  # Fijo
DROPOUT2 = 0.2  # Fijo


# =========================
# Definición del modelo (arquitectura fija, variar optimizador y dropout)
# =========================
def build_model(
    input_dim,
    optimizer_type="adam",
    lr=0.001,
    dropout1=0.3,
    dropout2=0.2,
    l2_reg=0.0001,
):
    """
    Construye el modelo con arquitectura fija.

    Args:
        input_dim: Dimensión de entrada
        optimizer_type: "gd" (Gradient Descent), "sgd" (Stochastic GD), "adam"
        lr: Learning rate
        dropout1: Dropout después de capa 1
        dropout2: Dropout después de capa 2
        l2_reg: Regularización L2
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(
                UNITS1,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name="dense_1",
            ),
            tf.keras.layers.Dropout(dropout1),
            tf.keras.layers.Dense(
                UNITS2,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name="dense_2",
            ),
            tf.keras.layers.Dropout(dropout2),
            tf.keras.layers.Dense(1, activation="sigmoid", name="out"),
        ]
    )

    # Seleccionar optimizador
    if optimizer_type.lower() == "gd":
        # Gradiente Descendente: SGD con batch_size completo (full batch)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_type.lower() == "sgd":
        # Descenso de Gradiente Estocástico: SGD con batch_size pequeño
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_type.lower() == "adam":
        # Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not supported")

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
        ],
    )
    return model


# =========================
# GRID SEARCH: Variar L2, learning_rate y épocas (ADAM - mejor para sin privacidad)
# IMPORTANTE: Usar validación split del train, NO el test
# Optimización por val_loss (mínimo)
# Threshold fijo: 0.2
# =========================
OPTIMIZER = "adam"  # Adam funciona mejor para modelos sin privacidad diferencial
THRESHOLD = 0.2  # Fijo
# Rangos expandidos para mejor búsqueda
L2_REG_CANDIDATES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3]  # Expandido: 6 valores
LEARNING_RATE_CANDIDATES = [
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
]  # Expandido: 6 valores (rangos para Adam)
EPOCHS_CANDIDATES = [
    30,
    40,
    50,
    60,
    80,
    100,
]  # Expandido: más épocas para mejor convergencia
BATCH_SIZE = 1024
VAL_SPLIT = 0.2  # Usar 20% del train para validación

# Callbacks - optimizar por val_loss (más permisivo para permitir mejor entrenamiento)
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,  # Aumentado de 7 a 15 para permitir más entrenamiento
    mode="min",
    restore_best_weights=True,
    verbose=0,
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=5,  # Aumentado de 3 a 5
    min_lr=1e-7,  # Más bajo para permitir más ajuste fino
    verbose=0,
)

results = []
# Optimizar por val_loss (mínimo)
best = {
    "val_loss": np.inf,
    "l2_reg": None,
    "learning_rate": None,
    "epochs": None,
}

# Imprimir resumen de hardware
hw_summary = hardware_summary()
print_hardware_summary(hw_summary)

print("\n" + "=" * 60)
print(f"BÚSQUEDA DE HIPERPARÁMETROS (SIN PRIVACIDAD DIFERENCIAL)")
print("=" * 60)
print(f"Arquitectura fija: Dense({UNITS1}) -> Dense({UNITS2}) -> Dense(1)")
print(f"Dropout fijo: {DROPOUT1} / {DROPOUT2}")
print(f"Optimizador: {OPTIMIZER.upper()} (fijo) - MEJOR PARA SIN PRIVACIDAD")
print(f"Threshold: {THRESHOLD} (fijo)")
print(f"L2 regularization a probar: {L2_REG_CANDIDATES}")
print(f"Learning rate a probar: {LEARNING_RATE_CANDIDATES}")
print(f"Épocas: {EPOCHS_CANDIDATES}")
total_combinations = (
    len(L2_REG_CANDIDATES) * len(LEARNING_RATE_CANDIDATES) * len(EPOCHS_CANDIDATES)
)
print(
    f"Total combinaciones: {len(L2_REG_CANDIDATES)} L2 x "
    f"{len(LEARNING_RATE_CANDIDATES)} LR x {len(EPOCHS_CANDIDATES)} ep = {total_combinations}"
)
print("=" * 60)

# =========================
# Configurar MLflow
# =========================
try:
    mlflow.set_experiment("DP-Fraud-Detection-Final")
except Exception as e:
    print(f"Advertencia al configurar experimento MLflow: {e}")
    print("Continuando con el experimento por defecto...")
    pass

# Run principal para el grid search
with mlflow.start_run(run_name="Grid_Search_No_DP_Optimized"):
    mlflow.log_param("architecture_units1", UNITS1)
    mlflow.log_param("architecture_units2", UNITS2)
    mlflow.log_param("optimizer", OPTIMIZER.upper())
    mlflow.log_param("dropout1", DROPOUT1)
    mlflow.log_param("dropout2", DROPOUT2)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("val_split", VAL_SPLIT)

    # Log hardware resources
    mlflow.log_param("num_cpus", hw_summary["logical_cpus"])
    mlflow.log_param("tf_intra_threads", hw_summary["tf_intra_threads"])
    mlflow.log_param("tf_inter_threads", hw_summary["tf_inter_threads"])
    if hw_summary.get("ram_total_bytes"):
        mlflow.log_param(
            "ram_total_gb", round(hw_summary["ram_total_bytes"] / (1024**3), 2)
        )
    if hw_summary.get("ram_available_bytes"):
        mlflow.log_param(
            "ram_available_gb", round(hw_summary["ram_available_bytes"] / (1024**3), 2)
        )

    # Log información de selección de variables
    mlflow.log_param("feature_selection_method", "VarianceThreshold+MutualInfo")
    mlflow.log_param("feature_variance_threshold", variance_threshold)
    mlflow.log_param("features_before_selection", len(feature_names))
    mlflow.log_param("features_after_variance", int(len(feature_names_after_var)))
    mlflow.log_param("features_selected", int(len(selected_feature_names)))
    mlflow.log_param("varsel_min_features", min_features)
    mlflow.log_param("varsel_max_features", max_features)

    # Guardar lista de características seleccionadas como artifact
    selected_feature_report = "\n".join(
        f"{name}: {score:.6f}"
        for name, score in zip(selected_feature_names, selected_mi_scores)
    )
    mlflow.log_text(selected_feature_report, "selected_features.txt")

    # Crear todas las combinaciones para la barra de progreso
    all_combinations = [
        (l2_reg, lr, ep)
        for l2_reg in L2_REG_CANDIDATES
        for lr in LEARNING_RATE_CANDIDATES
        for ep in EPOCHS_CANDIDATES
    ]

    # Barra de progreso
    pbar = tqdm(
        all_combinations,
        desc="Grid Search",
        unit="trial",
        total=len(all_combinations),
    )

    for l2_reg, lr, ep in pbar:
        # Run anidado para cada trial
        run_name = f"l2={l2_reg:.0e}_lr={lr:.0e}_ep={ep}"
        with mlflow.start_run(run_name=run_name, nested=True):
            # Log parámetros del trial
            mlflow.log_param("l2_regularization", l2_reg)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", ep)
            mlflow.log_param("threshold", THRESHOLD)

            model = build_model(
                X_train.shape[1],
                optimizer_type=OPTIMIZER,
                lr=lr,
                dropout1=DROPOUT1,
                dropout2=DROPOUT2,
                l2_reg=l2_reg,
            )

            mlflow.log_param("batch_size", BATCH_SIZE)

            hist = model.fit(
                X_train,
                y_train,
                epochs=ep,
                batch_size=BATCH_SIZE,
                validation_split=VAL_SPLIT,
                callbacks=[stop_early, reduce_lr],
                verbose=0,
            )

            # Log métricas por epoch
            for epoch_idx, (
                train_loss_ep,
                val_loss_ep,
                train_acc_ep,
                val_acc_ep,
            ) in enumerate(
                zip(
                    hist.history["loss"],
                    hist.history["val_loss"],
                    hist.history["accuracy"],
                    hist.history["val_accuracy"],
                ),
                start=1,
            ):
                mlflow.log_metric("train_loss", float(train_loss_ep), step=epoch_idx)
                mlflow.log_metric("val_loss", float(val_loss_ep), step=epoch_idx)
                mlflow.log_metric(
                    "train_accuracy",
                    float(train_acc_ep),
                    step=epoch_idx,
                )
                mlflow.log_metric("val_accuracy", float(val_acc_ep), step=epoch_idx)

                # Log otras métricas si están disponibles
                if "val_auprc" in hist.history:
                    mlflow.log_metric(
                        "val_auprc",
                        float(hist.history["val_auprc"][epoch_idx - 1]),
                        step=epoch_idx,
                    )
                if "val_auroc" in hist.history:
                    mlflow.log_metric(
                        "val_auroc",
                        float(hist.history["val_auroc"][epoch_idx - 1]),
                        step=epoch_idx,
                    )

            # Seleccionar mejor epoch por val_loss (mínimo)
            val_loss_history = hist.history["val_loss"]
            best_epoch_idx = int(np.argmin(val_loss_history))
            best_val_loss = float(val_loss_history[best_epoch_idx])

            val_acc = float(hist.history["val_accuracy"][best_epoch_idx])
            train_acc = float(hist.history["accuracy"][best_epoch_idx])
            train_loss = float(hist.history["loss"][best_epoch_idx])

            # Evaluar con threshold fijo usando el conjunto de validación
            # Obtener datos de validación para evaluar con threshold
            X_val_split, y_val_split = (
                X_train[int(len(X_train) * (1 - VAL_SPLIT)) :],
                y_train[int(len(y_train) * (1 - VAL_SPLIT)) :],
            )
            y_val_proba = model.predict(X_val_split, verbose=0).ravel()
            y_val_pred_thr = (y_val_proba >= THRESHOLD).astype(int)

            # Calcular métricas con threshold fijo
            prec_thr, rec_thr, f1_thr, _ = precision_recall_fscore_support(
                y_val_split,
                y_val_pred_thr,
                average=None,
                zero_division=0,
            )

            # Log métricas del mejor epoch como métricas finales del trial
            mlflow.log_metric("best_epoch", best_epoch_idx + 1)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_val_accuracy", val_acc)
            mlflow.log_metric("train_loss_best_epoch", train_loss)
            mlflow.log_metric("train_accuracy_best_epoch", train_acc)
            mlflow.log_metric("val_precision_class_1_threshold", prec_thr[1])
            mlflow.log_metric("val_recall_class_1_threshold", rec_thr[1])
            mlflow.log_metric("val_f1_class_1_threshold", f1_thr[1])

            # Guardar resultados
            results.append(
                {
                    "l2_reg": l2_reg,
                    "learning_rate": lr,
                    "epochs": ep,
                    "best_epoch": best_epoch_idx + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "val_precision_1": prec_thr[1],
                    "val_recall_1": rec_thr[1],
                    "val_f1_1": f1_thr[1],
                }
            )

            # Actualizar barra de progreso
            pbar.set_postfix(
                {
                    "L2": f"{l2_reg:.0e}",
                    "LR": f"{lr:.0e}",
                    "Ep": ep,
                    "Best_Val_Loss": f"{best_val_loss:.4f}",
                }
            )

            # Actualizar mejor si mejora val_loss (mínimo)
            if best_val_loss < best["val_loss"]:
                best.update(
                    {
                        "val_loss": best_val_loss,
                        "l2_reg": l2_reg,
                        "learning_rate": lr,
                        "epochs": ep,
                    }
                )

    # Log mejor combinación encontrada al finalizar grid search
    mlflow.log_param("best_l2_reg", best["l2_reg"])
    mlflow.log_param("best_learning_rate", best["learning_rate"])
    mlflow.log_param("best_epochs", best["epochs"])
    mlflow.log_metric("best_val_loss_found", best["val_loss"])

    # Convertir resultados a DataFrame y guardar CSV ANTES de loguearlo
    results_df = pd.DataFrame(results)
    results_df.to_csv("optimizer_search_results.csv", index=False)

    # Guardar CSV de resultados como artifact (solo si existe)
    if os.path.exists("optimizer_search_results.csv"):
        mlflow.log_artifact("optimizer_search_results.csv")


# Ordenar resultados por val_loss (ascendente) - results_df ya fue creado arriba
results_sorted = results_df.sort_values("val_loss", ascending=True)

print("\n" + "=" * 60)
print("=== Grid Search Summary (sorted by val_loss - mejor primero) ===")
print("=" * 60)
print(
    f"{'L2':<10} {'LR':<10} {'Epochs':<8} {'Best_Ep':<10} {'Val_Loss':<12} {'Val_Acc':<12} {'F1_1':<8}"
)
print("-" * 75)
for _, row in results_sorted.head(15).iterrows():
    print(
        f"{row['l2_reg']:<10.0e} {row['learning_rate']:<10.0e} "
        f"{row['epochs']:<8} {row['best_epoch']:<10} "
        f"{row['val_loss']:<12.4f} {row['val_acc']:<12.4f} {row['val_f1_1']:<8.4f}"
    )

# CSV ya fue guardado arriba antes del logueo en MLflow

print("\n" + "=" * 60)
print("MEJOR COMBINACIÓN ENCONTRADA:")
print("=" * 60)
print(f"Optimizador: {OPTIMIZER.upper()} (fijo)")
print(f"L2 regularization: {best['l2_reg']:.0e}")
print(f"Learning rate: {best['learning_rate']:.0e}")
print(f"Épocas: {best['epochs']}")
print(f"Threshold: {THRESHOLD} (fijo)")
print(f"Val Loss: {best['val_loss']:.4f}")
print("=" * 60)

# =========================
# Re-entrenar con la mejor combinación y evaluar en TEST
# =========================
print("\n" + "=" * 60)
print("ENTRENAMIENTO FINAL CON MEJOR OPTIMIZADOR")
print("=" * 60)

# Run final para entrenamiento y evaluación
with mlflow.start_run(run_name="FINAL_Best_Model"):
    # Log parámetros finales
    mlflow.log_param("optimizer", OPTIMIZER.upper())
    mlflow.log_param("dropout1", DROPOUT1)
    mlflow.log_param("dropout2", DROPOUT2)
    mlflow.log_param("epochs", best["epochs"])
    mlflow.log_param("learning_rate", best["learning_rate"])
    mlflow.log_param("l2_regularization", best["l2_reg"])
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("architecture_units1", UNITS1)
    mlflow.log_param("architecture_units2", UNITS2)

    best_model = build_model(
        X_train.shape[1],
        optimizer_type=OPTIMIZER,
        lr=best["learning_rate"],
        dropout1=DROPOUT1,
        dropout2=DROPOUT2,
        l2_reg=best["l2_reg"],
    )

    mlflow.log_param("batch_size", BATCH_SIZE)

    history_final = best_model.fit(
        X_train,
        y_train,
        epochs=best["epochs"],
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[stop_early, reduce_lr],
        verbose=1,
    )

    # Log métricas del entrenamiento final
    for epoch_idx, (train_loss_ep, val_loss_ep, train_acc_ep, val_acc_ep) in enumerate(
        zip(
            history_final.history["loss"],
            history_final.history["val_loss"],
            history_final.history["accuracy"],
            history_final.history["val_accuracy"],
        ),
        start=1,
    ):
        mlflow.log_metric("final_train_loss", float(train_loss_ep), step=epoch_idx)
        mlflow.log_metric("final_val_loss", float(val_loss_ep), step=epoch_idx)
        mlflow.log_metric("final_train_accuracy", float(train_acc_ep), step=epoch_idx)
        mlflow.log_metric("final_val_accuracy", float(val_acc_ep), step=epoch_idx)

    # =========================
    # Evaluación en TEST (no usado durante búsqueda)
    # =========================
    test_metrics = best_model.evaluate(X_test, y_test, verbose=0)
    # evaluate devuelve: [loss, accuracy, precision, recall, auroc, auprc]
    test_loss = test_metrics[0]
    test_acc = test_metrics[1]
    test_precision = test_metrics[2]
    test_recall = test_metrics[3]
    test_auroc = test_metrics[4]
    test_auprc = test_metrics[5]

    print(f"\n>>> EVALUACIÓN EN TEST (conjunto no visto durante búsqueda):")
    print(f"    Test loss: {test_loss:.4f}")
    print(f"    Test accuracy: {test_acc:.4f}")
    print(f"    Test precision: {test_precision:.4f}")
    print(f"    Test recall: {test_recall:.4f}")
    print(f"    Test AUROC: {test_auroc:.4f}")
    print(f"    Test AUPRC: {test_auprc:.4f}")

    # Log métricas de test
    mlflow.log_metric("test_loss", float(test_loss))
    mlflow.log_metric("test_accuracy", float(test_acc))
    mlflow.log_metric("test_precision", float(test_precision))
    mlflow.log_metric("test_recall", float(test_recall))
    mlflow.log_metric("test_auroc", float(test_auroc))
    mlflow.log_metric("test_auprc", float(test_auprc))

    # Predicciones
    y_pred_prob = best_model.predict(X_test, verbose=0).ravel()

    # =========================
    # Evaluación con threshold fijo (0.2)
    # =========================
    y_pred = (y_pred_prob >= THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print(f"MÉTRICAS CON THRESHOLD FIJO ({THRESHOLD}):")
    print("=" * 60)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Log métricas con threshold fijo
    mlflow.log_metric("test_precision_class_0", report["0"]["precision"])
    mlflow.log_metric("test_recall_class_0", report["0"]["recall"])
    mlflow.log_metric("test_f1_class_0", report["0"]["f1-score"])
    mlflow.log_metric("test_precision_class_1", report["1"]["precision"])
    mlflow.log_metric("test_recall_class_1", report["1"]["recall"])
    mlflow.log_metric("test_f1_class_1", report["1"]["f1-score"])
    mlflow.log_metric("test_precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("test_recall_macro", report["macro avg"]["recall"])
    mlflow.log_metric("test_f1_macro", report["macro avg"]["f1-score"])

    # Guardar matriz de confusión
    cm_df = pd.DataFrame(
        cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
    )
    cm_path = "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    # =========================
    # RESUMEN FINAL DEL MEJOR MODELO
    # =========================
    print("\n" + "=" * 80)
    print("=" * 80)
    print("MEJOR MODELO ENCONTRADO (SIN PRIVACIDAD DIFERENCIAL - OPTIMIZADO)")
    print("=" * 80)
    print("=" * 80)
    print("\n📊 HIPERPARÁMETROS SELECCIONADOS:")
    print("  • Arquitectura: Dense({}) -> Dense({}) -> Dense(1)".format(UNITS1, UNITS2))
    print("  • Dropout capa 1/2: {:.2f} / {:.2f}".format(DROPOUT1, DROPOUT2))
    print("  • Optimizador: {}".format(OPTIMIZER.upper()))
    print("  • L2 regularization: {:.0e}".format(best["l2_reg"]))
    print("  • Learning Rate: {:.0e}".format(best["learning_rate"]))
    print("  • Épocas: {}".format(best["epochs"]))
    print("  • Batch Size: {}".format(BATCH_SIZE))
    print("  • Threshold: {}".format(THRESHOLD))
    print("  • Características seleccionadas: {}".format(len(selected_feature_names)))

    print("\n📈 MÉTRICAS EN VALIDACIÓN (mejor epoch):")
    print("  • Val Loss: {:.4f}".format(best["val_loss"]))

    print("\n🎯 MÉTRICAS EN TEST (conjunto no visto):")
    print("  • Test Loss: {:.4f}".format(test_loss))
    print("  • Test Accuracy: {:.4f}".format(test_acc))
    print("  • Test Precision: {:.4f}".format(test_precision))
    print("  • Test Recall: {:.4f}".format(test_recall))
    print("  • Test AUROC: {:.4f}".format(test_auroc))
    print("  • Test AUPRC: {:.4f}".format(test_auprc))

    print("\n📋 MÉTRICAS CON THRESHOLD FIJO ({:.2f}):".format(THRESHOLD))
    print("  • Precision (Clase 0): {:.4f}".format(report["0"]["precision"]))
    print("  • Recall (Clase 0): {:.4f}".format(report["0"]["recall"]))
    print("  • F1 (Clase 0): {:.4f}".format(report["0"]["f1-score"]))
    print("  • Precision (Clase 1): {:.4f}".format(report["1"]["precision"]))
    print("  • Recall (Clase 1): {:.4f}".format(report["1"]["recall"]))
    print("  • F1 (Clase 1): {:.4f}".format(report["1"]["f1-score"]))
    print("  • Precision (Macro): {:.4f}".format(report["macro avg"]["precision"]))
    print("  • Recall (Macro): {:.4f}".format(report["macro avg"]["recall"]))
    print("  • F1 (Macro): {:.4f}".format(report["macro avg"]["f1-score"]))

    print("\n🔢 MATRIZ DE CONFUSIÓN:")
    print(cm_df)

    print("\n" + "=" * 80)
    print("=" * 80)

"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
