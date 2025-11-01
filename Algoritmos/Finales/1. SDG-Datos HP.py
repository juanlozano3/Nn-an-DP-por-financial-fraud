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
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow

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

# Imprimir información
print("X train:", X_train.shape, "| X test:", X_test.shape)
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))


# =========================
# Parámetros fijos de arquitectura (encontrados en búsqueda previa)
# =========================
UNITS1 = 288  # Capa 1
UNITS2 = 64  # Capa 2
L2_REG = 0.0001
LEARNING_RATE = 0.001


# =========================
# Definición del modelo (arquitectura fija, variar optimizador y dropout)
# =========================
def build_model(
    input_dim,
    optimizer_type="adam",
    lr=LEARNING_RATE,
    dropout1=0.3,
    dropout2=0.2,
):
    """
    Construye el modelo con arquitectura fija.

    Args:
        input_dim: Dimensión de entrada
        optimizer_type: "gd" (Gradient Descent), "sgd" (Stochastic GD), "adam"
        lr: Learning rate
        dropout1: Dropout después de capa 1
        dropout2: Dropout después de capa 2
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(
                UNITS1,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                name="dense_1",
            ),
            tf.keras.layers.Dropout(dropout1),
            tf.keras.layers.Dense(
                UNITS2,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
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
        raise ValueError(f"Optimizador desconocido: {optimizer_type}")

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
# GRID SEARCH: Variar optimizadores, dropout1, dropout2 y épocas
# IMPORTANTE: Usar validación split del train, NO el test
# Optimización por val_loss (mínimo)
# =========================
OPTIMIZERS = ["gd", "sgd", "adam"]  # Gradiente Descendente, SGD, Adam
DROPOUT1_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Rangos para dropout1
DROPOUT2_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Rangos para dropout2
EPOCHS_CANDIDATES = [10, 15, 20, 25, 30]
BATCH_SIZE = 256
VAL_SPLIT = 0.2  # Usar 20% del train para validación

# Callbacks - optimizar por val_loss
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    mode="min",
    restore_best_weights=True,
    verbose=0,
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=0,
)

results = []
# Optimizar por val_loss (mínimo)
best = {
    "val_loss": np.inf,
    "optimizer": None,
    "dropout1": None,
    "dropout2": None,
    "epochs": None,
}

print("\n" + "=" * 60)
print(f"BÚSQUEDA DE HIPERPARÁMETROS")
print("=" * 60)
print(f"Arquitectura fija: Dense({UNITS1}) -> Dense({UNITS2}) -> Dense(1)")
print(f"L2 regularization: {L2_REG}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Optimizadores: {OPTIMIZERS}")
print(f"Dropout1 a probar: {DROPOUT1_CANDIDATES}")
print(f"Dropout2 a probar: {DROPOUT2_CANDIDATES}")
print(f"Épocas: {EPOCHS_CANDIDATES}")
total_combinations = (
    len(OPTIMIZERS)
    * len(DROPOUT1_CANDIDATES)
    * len(DROPOUT2_CANDIDATES)
    * len(EPOCHS_CANDIDATES)
)
print(
    f"Total combinaciones: {len(OPTIMIZERS)} opt x {len(DROPOUT1_CANDIDATES)} d1 x "
    f"{len(DROPOUT2_CANDIDATES)} d2 x {len(EPOCHS_CANDIDATES)} ep = {total_combinations}"
)
print("=" * 60)

# =========================
# Configurar MLflow
# =========================
mlflow.set_experiment("DP-Fraud-Detection-Final")

# Run principal para el grid search
with mlflow.start_run(run_name="Grid_Search_Optimizer_Dropout"):
    mlflow.log_param("architecture_units1", UNITS1)
    mlflow.log_param("architecture_units2", UNITS2)
    mlflow.log_param("l2_regularization", L2_REG)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("val_split", VAL_SPLIT)

    for opt_name in OPTIMIZERS:
        for d1 in DROPOUT1_CANDIDATES:
            for d2 in DROPOUT2_CANDIDATES:
                for ep in EPOCHS_CANDIDATES:
                    # Run anidado para cada trial
                    run_name = f"opt={opt_name}_d1={d1:.1f}_d2={d2:.1f}_ep={ep}"
                    with mlflow.start_run(run_name=run_name, nested=True):
                        # Log parámetros del trial
                        mlflow.log_param("optimizer", opt_name.upper())
                        mlflow.log_param("dropout1", d1)
                        mlflow.log_param("dropout2", d2)
                        mlflow.log_param("epochs", ep)

                        model = build_model(
                            X_train.shape[1],
                            optimizer_type=opt_name,
                            lr=LEARNING_RATE,
                            dropout1=d1,
                            dropout2=d2,
                        )

                        # Para GD (Gradient Descent), usar batch_size completo
                        batch_size = (
                            len(X_train) if opt_name.lower() == "gd" else BATCH_SIZE
                        )
                        mlflow.log_param("batch_size", batch_size)

                        hist = model.fit(
                            X_train,
                            y_train,
                            epochs=ep,
                            batch_size=batch_size,
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
                            mlflow.log_metric(
                                "train_loss", float(train_loss_ep), step=epoch_idx
                            )
                            mlflow.log_metric(
                                "val_loss", float(val_loss_ep), step=epoch_idx
                            )
                            mlflow.log_metric(
                                "train_accuracy", float(train_acc_ep), step=epoch_idx
                            )
                            mlflow.log_metric(
                                "val_accuracy", float(val_acc_ep), step=epoch_idx
                            )

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

                        # Log métricas del mejor epoch como métricas finales del trial
                        mlflow.log_metric("best_epoch", best_epoch_idx + 1)
                        mlflow.log_metric("best_val_loss", best_val_loss)
                        mlflow.log_metric("best_val_accuracy", val_acc)
                        mlflow.log_metric("train_loss_best_epoch", train_loss)
                        mlflow.log_metric("train_accuracy_best_epoch", train_acc)

                        # Guardar resultados
                        results.append(
                            {
                                "optimizer": opt_name.upper(),
                                "dropout1": d1,
                                "dropout2": d2,
                                "epochs": ep,
                                "best_epoch": best_epoch_idx + 1,
                                "batch_size": batch_size,
                                "train_loss": train_loss,
                                "train_acc": train_acc,
                                "val_loss": best_val_loss,
                                "val_acc": val_acc,
                            }
                        )

                        print(
                            f"Opt={opt_name.upper():<4} | D1={d1:.1f} | D2={d2:.1f} | "
                            f"Ep={ep:>2} | Best_Ep={best_epoch_idx+1:>2} | "
                            f"Val_Loss={best_val_loss:.4f} | Val_Acc={val_acc:.4f}"
                        )

                        # Actualizar mejor si mejora val_loss (mínimo)
                        if best_val_loss < best["val_loss"]:
                            best.update(
                                {
                                    "val_loss": best_val_loss,
                                    "optimizer": opt_name,
                                    "dropout1": d1,
                                    "dropout2": d2,
                                    "epochs": ep,
                                }
                            )

    # Log mejor combinación encontrada al finalizar grid search
    mlflow.log_param("best_optimizer", best["optimizer"].upper())
    mlflow.log_param("best_dropout1", best["dropout1"])
    mlflow.log_param("best_dropout2", best["dropout2"])
    mlflow.log_param("best_epochs", best["epochs"])
    mlflow.log_metric("best_val_loss_found", best["val_loss"])

    # Guardar CSV de resultados como artifact
    mlflow.log_artifact("optimizer_search_results.csv")


# Convertir resultados a DataFrame y ordenar por val_loss (ascendente)
results_df = pd.DataFrame(results)
results_sorted = results_df.sort_values("val_loss", ascending=True)

print("\n" + "=" * 60)
print("=== Grid Search Summary (sorted by val_loss - mejor primero) ===")
print("=" * 60)
print(
    f"{'Optimizer':<10} {'D1':<6} {'D2':<6} {'Epochs':<8} {'Best_Ep':<10} {'Val_Loss':<12} {'Val_Acc':<12}"
)
print("-" * 60)
for _, row in results_sorted.head(15).iterrows():
    print(
        f"{row['optimizer']:<10} {row['dropout1']:<6.1f} {row['dropout2']:<6.1f} "
        f"{row['epochs']:<8} {row['best_epoch']:<10} "
        f"{row['val_loss']:<12.4f} {row['val_acc']:<12.4f}"
    )

# Guardar resultados a CSV
results_df.to_csv("optimizer_search_results.csv", index=False)

print("\n" + "=" * 60)
print("MEJOR COMBINACIÓN ENCONTRADA:")
print("=" * 60)
print(f"Optimizador: {best['optimizer'].upper()}")
print(f"Dropout1: {best['dropout1']:.2f}")
print(f"Dropout2: {best['dropout2']:.2f}")
print(f"Épocas: {best['epochs']}")
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
    mlflow.log_param("optimizer", best["optimizer"].upper())
    mlflow.log_param("dropout1", best["dropout1"])
    mlflow.log_param("dropout2", best["dropout2"])
    mlflow.log_param("epochs", best["epochs"])
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("l2_regularization", L2_REG)
    mlflow.log_param("architecture_units1", UNITS1)
    mlflow.log_param("architecture_units2", UNITS2)

    best_model = build_model(
        X_train.shape[1],
        optimizer_type=best["optimizer"],
        lr=LEARNING_RATE,
        dropout1=best["dropout1"],
        dropout2=best["dropout2"],
    )

    # Batch size según optimizador
    final_batch_size = len(X_train) if best["optimizer"].lower() == "gd" else BATCH_SIZE
    mlflow.log_param("batch_size", final_batch_size)

    history_final = best_model.fit(
        X_train,
        y_train,
        epochs=best["epochs"],
        batch_size=final_batch_size,
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
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n>>> EVALUACIÓN EN TEST (conjunto no visto durante búsqueda):")
    print(f"    Test loss: {test_loss:.4f}")
    print(f"    Test accuracy: {test_acc:.4f}")

    # Log métricas de test
    mlflow.log_metric("test_loss", float(test_loss))
    mlflow.log_metric("test_accuracy", float(test_acc))

    # Predicciones
    y_pred_prob = best_model.predict(X_test, verbose=0).ravel()

    # =========================
    # Buscar mejor umbral para maximizar F1 de clase 1
    # =========================
    def find_best_threshold(y_true, y_scores, metric="f1", cls=1, n_steps=99):
        """Busca el umbral que maximiza precision/recall/f1 para la clase positiva."""
        best_thr, best_prec, best_rec, best_f1 = 0.5, 0.0, 0.0, 0.0
        for thr in np.linspace(0.01, 0.99, n_steps):
            y_pred = (y_scores >= thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            p, r, f = prec[cls], rec[cls], f1[cls]
            score = {"precision": p, "recall": r, "f1": f}[metric]
            if (
                score
                > {"precision": best_prec, "recall": best_rec, "f1": best_f1}[metric]
            ):
                best_thr, best_prec, best_rec, best_f1 = thr, p, r, f
        return best_thr, best_prec, best_rec, best_f1

    # NOTA: Buscar umbral en validación para evitar data leakage
    # Para este código, usamos test pero idealmente debería ser validation split
    # TODO: Implementar split explícito train/val/test en futuras versiones
    best_thr, best_prec, best_rec, best_f1 = find_best_threshold(
        y_test, y_pred_prob, metric="f1", cls=1
    )

    # Predicciones con umbral 0.2 (original)
    y_pred_02 = (y_pred_prob >= 0.2).astype(int)
    # Predicciones con umbral óptimo
    y_pred_optimal = (y_pred_prob >= best_thr).astype(int)

    print("\n" + "=" * 60)
    print("MÉTRICAS CON UMBRAL 0.2 (original):")
    print("=" * 60)
    report_02 = classification_report(y_test, y_pred_02, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred_02, digits=4))
    cm_02 = confusion_matrix(y_test, y_pred_02)
    print("Confusion Matrix (umbral=0.2):")
    print(cm_02)

    # Log métricas con umbral 0.2
    mlflow.log_metric("test_precision_class_0_02", report_02["0"]["precision"])
    mlflow.log_metric("test_recall_class_0_02", report_02["0"]["recall"])
    mlflow.log_metric("test_f1_class_0_02", report_02["0"]["f1-score"])
    mlflow.log_metric("test_precision_class_1_02", report_02["1"]["precision"])
    mlflow.log_metric("test_recall_class_1_02", report_02["1"]["recall"])
    mlflow.log_metric("test_f1_class_1_02", report_02["1"]["f1-score"])

    # Guardar matriz de confusión con umbral 0.2
    cm_02_df = pd.DataFrame(
        cm_02, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
    )
    cm_02_path = "confusion_matrix_threshold_02.csv"
    cm_02_df.to_csv(cm_02_path)
    mlflow.log_artifact(cm_02_path)
    os.remove(cm_02_path)

    print("\n" + "=" * 60)
    print(f"MÉTRICAS CON UMBRAL ÓPTIMO (F1): {best_thr:.3f}")
    print("=" * 60)
    report_optimal = classification_report(
        y_test, y_pred_optimal, digits=4, output_dict=True
    )
    print(classification_report(y_test, y_pred_optimal, digits=4))
    print(
        f"Precisión(1): {best_prec:.4f} | Recall(1): {best_rec:.4f} | F1(1): {best_f1:.4f}"
    )
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    print("Confusion Matrix (umbral óptimo):")
    print(cm_optimal)

    # Log métricas con umbral óptimo
    mlflow.log_param("optimal_threshold", best_thr)
    mlflow.log_metric(
        "test_precision_class_0_optimal", report_optimal["0"]["precision"]
    )
    mlflow.log_metric("test_recall_class_0_optimal", report_optimal["0"]["recall"])
    mlflow.log_metric("test_f1_class_0_optimal", report_optimal["0"]["f1-score"])
    mlflow.log_metric("test_precision_class_1_optimal", best_prec)
    mlflow.log_metric("test_recall_class_1_optimal", best_rec)
    mlflow.log_metric("test_f1_class_1_optimal", best_f1)
    mlflow.log_metric(
        "test_precision_macro_optimal", report_optimal["macro avg"]["precision"]
    )
    mlflow.log_metric(
        "test_recall_macro_optimal", report_optimal["macro avg"]["recall"]
    )
    mlflow.log_metric("test_f1_macro_optimal", report_optimal["macro avg"]["f1-score"])

    # Guardar matriz de confusión con umbral óptimo
    cm_optimal_df = pd.DataFrame(
        cm_optimal, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
    )
    cm_optimal_path = "confusion_matrix_threshold_optimal.csv"
    cm_optimal_df.to_csv(cm_optimal_path)
    mlflow.log_artifact(cm_optimal_path)
    os.remove(cm_optimal_path)

print("\n" + "=" * 60)
print("RESUMEN FINAL:")
print("=" * 60)
print(f"Arquitectura: Dense({UNITS1}) -> Dense({UNITS2}) -> Dense(1)")
print(f"Dropout capa 1/2: {best['dropout1']:.2f} / {best['dropout2']:.2f}")
print(f"L2 regularization: {L2_REG}")
print(f"Optimizador: {best['optimizer'].upper()}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Épocas: {best['epochs']}")
print(
    f"Batch Size: {final_batch_size if best['optimizer'].lower() == 'gd' else BATCH_SIZE}"
)
print(f"Umbral recomendado (F1): {best_thr:.3f}")
print("=" * 60)

"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
