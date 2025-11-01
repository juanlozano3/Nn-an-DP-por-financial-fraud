import os, random
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

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

# Dummies para categóricas
X = pd.get_dummies(X, drop_first=True)

# Split primero (evita data leakage con el scaler)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
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

X_majority = X_train_df[y_train_series == 0]
X_minority = X_train_df[y_train_series == 1]
y_majority = y_train_series[y_train_series == 0]
y_minority = y_train_series[y_train_series == 1]

X_majority_down, y_majority_down = resample(
    X_majority, y_majority, replace=False, n_samples=len(y_minority), random_state=SEED
)

X_train_bal = pd.concat([X_majority_down, X_minority], axis=0)
y_train_bal = pd.concat([y_majority_down, y_minority], axis=0)

# Barajar (shuffle) después de concatenar
shuf_idx = np.random.RandomState(SEED).permutation(len(y_train_bal))
X_train = X_train_bal.to_numpy().astype(np.float32)[shuf_idx]
y_train = y_train_bal.to_numpy().astype(np.int32)[shuf_idx]

# Asegurar tipos en test
X_test = X_test.astype(np.float32)
y_test = y_test.to_numpy().astype(np.int32)

print("X train:", X_train.shape, "| X test:", X_test.shape)
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))


# =========================
# Parámetros FIJOS (no se buscan)
# =========================
# Esta búsqueda SOLO optimiza las unidades por capa (units1, units2)
# Todos los demás parámetros son fijos. Puedes modificarlos aquí y luego
# hacer búsquedas separadas con diferentes optimizadores/LR/etc.
OPTIMIZER = "adam"  # Opciones: "adam", "sgd"
LEARNING_RATE = 0.001  # Ajustar según necesites
DROPOUT1 = 0.3  # Dropout después de la primera capa
DROPOUT2 = 0.2  # Dropout después de la segunda capa
L2_REG = 1e-4  # Regularización L2 en todas las capas


# =========================
# Hiperparámetros y modelo
# Solo busca: unidades por capa
# =========================
def model_builder(hp: kt.HyperParameters):
    model = keras.Sequential()
    # SOLO buscar unidades de las capas
    hp_units1 = hp.Int("units1", min_value=32, max_value=512, step=32)
    hp_units2 = hp.Int("units2", min_value=16, max_value=256, step=16)
    # Opcional: también buscar dropout si quieres, pero por ahora fijo
    # hp_drop1 = hp.Float("dropout1", 0.0, 0.5, step=0.1)
    # hp_drop2 = hp.Float("dropout2", 0.0, 0.5, step=0.1)

    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(
        keras.layers.Dense(
            units=hp_units1,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REG),
        )
    )
    model.add(keras.layers.Dropout(DROPOUT1))
    model.add(
        keras.layers.Dense(
            units=hp_units2,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(L2_REG),
        )
    )
    model.add(keras.layers.Dropout(DROPOUT2))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Optimizador y LR fijos
    if OPTIMIZER.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif OPTIMIZER.lower() == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auroc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
        ],
    )
    return model


# Tuner optimizando AUPRC (mejor para clase positiva rara)
# Hyperband es eficiente, pero aumenta max_epochs para mejor exploración
tuner = kt.Hyperband(
    model_builder,
    objective=kt.Objective("val_auprc", direction="max"),
    max_epochs=30,  # Aumentado para mejor convergencia
    factor=3,
    directory="my_dir",
    project_name="tuning_tabular_model",
    overwrite=True,
    executions_per_trial=1,  # Reducir tiempo, pero podrías aumentar para más robustez
)

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_auprc",
    patience=7,
    mode="max",
    restore_best_weights=True,  # Más paciencia
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auprc",
    factor=0.5,
    patience=3,
    mode="max",
    min_lr=1e-6,
    verbose=1,  # Más paciencia, LR mínimo más bajo
)

# Considera agregar batch_size como hiperparámetro, pero por ahora fijo
# Para DP-SGD, batch_size típicamente necesita ser mayor (256-512)
BATCH_SIZE = 256

tuner.search(
    X_train,
    y_train,
    epochs=50,
    batch_size=BATCH_SIZE,  # Explícito
    validation_split=0.2,
    callbacks=[stop_early, reduce_lr],
    verbose=1,
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "=" * 50)
print("  BÚSQUEDA DE ARQUITECTURA FINALIZADA")
print("=" * 50)
print(f"Mejor units1 (capa 1):  {best_hps.get('units1')}")
print(f"Mejor units2 (capa 2):  {best_hps.get('units2')}")
print(f"\nParámetros fijos usados:")
print(f"  Optimizer:            {OPTIMIZER}")
print(f"  Learning rate:        {LEARNING_RATE}")
print(f"  Dropout capa 1/2:     {DROPOUT1} / {DROPOUT2}")
print(f"  L2 regularization:   {L2_REG}")
print("=" * 50)

# === Construye el modelo con los mejores HPs ===
model = tuner.hypermodel.build(best_hps)

# Entrenamiento final con los mejores HPs
# Usa el mismo batch_size que en la búsqueda
history = model.fit(
    X_train,
    y_train,
    epochs=100,  # Más épocas para entrenamiento final
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[stop_early, reduce_lr],
    verbose=1,
)

# =========================
# Evaluación
# =========================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")

# Predicciones y métricas detalladas
y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nClassification Report (umbral=0.5):")
print(classification_report(y_test, y_pred, digits=4))


# =========================
# Recomendación de parámetros finales
# =========================
def find_best_threshold(y_true, y_scores, metric="f1", cls=1, n_steps=99):
    """
    Busca el umbral en [0,1] que maximiza precision/recall/f1
    para la clase positiva (cls=1).
    """
    best_thr, best_prec, best_rec, best_f1 = 0.5, 0.0, 0.0, 0.0
    for thr in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        p, r, f = prec[cls], rec[cls], f1[cls]
        score = {"precision": p, "recall": r, "f1": f}[metric]
        if score > {"precision": best_prec, "recall": best_rec, "f1": best_f1}[metric]:
            best_thr, best_prec, best_rec, best_f1 = thr, p, r, f
    return best_thr, best_prec, best_rec, best_f1


# 1) Umbral que maximiza F1 de la clase 1 (fraude)
best_thr, best_prec, best_rec, best_f1 = find_best_threshold(
    y_test, y_pred_prob, metric="f1", cls=1
)

# 2) Métricas y CM con ese umbral recomendado
y_pred_best = (y_pred_prob >= best_thr).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best)

# 3) Paquete final de parámetros - ARQUITECTURA
print("\n" + "=" * 50)
print("     ARQUITECTURA ENCONTRADA (SOLO CAPAS)")
print("=" * 50)
print(f"Units capa 1:          {best_hps.get('units1')}")
print(f"Units capa 2:          {best_hps.get('units2')}")
print(f"\nParámetros fijos usados en la búsqueda:")
print(f"  Optimizer:           {OPTIMIZER}")
print(f"  Learning rate:       {LEARNING_RATE}")
print(f"  Dropout capa 1/2:    {DROPOUT1} / {DROPOUT2}")
print(f"  L2 regularization:  {L2_REG}")
print(f"  Scaler:              StandardScaler(with_mean=True, with_std=True)")
print(f"  Balanceo:            Downsampling mayoritaria en train")
print(f"  Batch size:          {BATCH_SIZE}")
print(f"\nEvaluación en test (con arquitectura encontrada):")
print(f"  Umbral recomendado (F1 clase 1): {best_thr:.3f}")
print(
    f"  -> Precisión(1): {best_prec:.4f} | Recall(1): {best_rec:.4f} | F1(1): {best_f1:.4f}"
)
print("\nMatriz de confusión con umbral recomendado:")
print(cm_best)
