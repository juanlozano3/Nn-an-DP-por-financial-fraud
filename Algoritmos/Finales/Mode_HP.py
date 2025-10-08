import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

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
    X, y, test_size=0.2, random_state=42, stratify=y
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
    X_majority,
    y_majority,
    replace=False,
    n_samples=len(y_minority),
    random_state=42,
)

X_train_bal = pd.concat([X_majority_down, X_minority], axis=0)
y_train_bal = pd.concat([y_majority_down, y_minority], axis=0)

# Barajar (shuffle) después de concatenar
shuf_idx = np.random.RandomState(42).permutation(len(y_train_bal))
X_train = X_train_bal.to_numpy().astype(np.float32)[shuf_idx]
y_train = y_train_bal.to_numpy().astype(np.int32)[shuf_idx]

# Asegurar tipos en test
X_test = X_test.astype(np.float32)
y_test = y_test.to_numpy().astype(np.int32)

print("X train:", X_train.shape, "| X test:", X_test.shape)
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))


# =========================
# Hiperparámetros y modelo
# =========================
def model_builder(hp: kt.HyperParameters):
    model = keras.Sequential()
    hp_units1 = hp.Int("units1", min_value=32, max_value=512, step=32)
    hp_units2 = hp.Int("units2", min_value=32, max_value=256, step=32)

    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(keras.layers.Dense(units=hp_units1, activation="relu"))
    model.add(keras.layers.Dense(units=hp_units2, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    hp_lr = hp.Choice("learning_rate", values=[0.01, 0.05, 0.1, 0.25])
    optimizer = keras.optimizers.SGD(learning_rate=hp_lr)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


tuner = kt.Hyperband(
    model_builder,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="my_dir",
    project_name="tuning_tabular_model",
)

stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

tuner.search(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[stop_early],
    verbose=1,
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(
    f"\nHPO finalizado.\n"
    f"Mejor units1: {best_hps.get('units1')}\n"
    f"Mejor units2: {best_hps.get('units2')}\n"
    f"Mejor learning_rate: {best_hps.get('learning_rate')}\n"
)

# === AQUÍ SE CONSTRUYE EL MODELO CON LOS MEJORES HPs (evita NameError) ===
model = tuner.hypermodel.build(best_hps)

# Entrenamiento final con los mejores HPs
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[stop_early],
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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def find_best_threshold(y_true, y_scores, metric="f1", cls=1, n_steps=99):
    """
    Busca el umbral en [0,1] que maximiza la métrica indicada (precision/recall/f1)
    sobre la clase positiva (cls=1).
    """
    best_thr, best_prec, best_rec, best_f1 = 0.5, 0.0, 0.0, 0.0
    for thr in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_scores >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        # índice de la clase cls
        p, r, f = prec[cls], rec[cls], f1[cls]
        score = {"precision": p, "recall": r, "f1": f}[metric]
        if score > {"precision": best_prec, "recall": best_rec, "f1": best_f1}[metric]:
            best_thr, best_prec, best_rec, best_f1 = thr, p, r, f
    return best_thr, best_prec, best_rec, best_f1


# 1) Encontrar umbral que maximiza F1 de la clase 1 (fraude)
best_thr, best_prec, best_rec, best_f1 = find_best_threshold(
    y_test, y_pred_prob, metric="f1", cls=1
)

# 2) Métricas y CM con ese umbral recomendado
y_pred_best = (y_pred_prob >= best_thr).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best)

# 3) Imprimir "paquete" de parámetros recomendados
print("\n============================")
print("     PARÁMETROS A USAR      ")
print("============================")
print(f"Mejor units1:        {best_hps.get('units1')}")
print(f"Mejor units2:        {best_hps.get('units2')}")
print(f"Mejor learning_rate: {best_hps.get('learning_rate')}")
print(f"Scaler:              StandardScaler(with_mean=True, with_std=True)")
print(f"Balanceo:            Downsampling mayoritaria en train")
print(f"Umbral recomendado (F1 clase 1): {best_thr:.3f}")
print(
    f"-> Precisión(1): {best_prec:.4f} | Recall(1): {best_rec:.4f} | F1(1): {best_f1:.4f}"
)
print("Matriz de confusión con umbral recomendado:")
print(cm_best)
