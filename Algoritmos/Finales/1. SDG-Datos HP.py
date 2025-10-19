import numpy as np
import pandas as pd
from sklearn.utils import resample
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("../../Datos/2/Base.csv")

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

# Convert the categorical columns to dummies
X = pd.get_dummies(X, drop_first=True)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

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
shuf = np.random.RandomState(42).permutation(len(y_train))
X_train = X_train[shuf]
y_train = y_train[shuf]
# 6. Imprimir las nuevas formas
print("X train: ", len(X_train))
print("X Test: ", len(X_test))
print("y Train: ", len(y_train))
print("y Test: ", len(y_test))
# Después del balanceo
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))


# =========================
# Definición del modelo (misma arquitectura)
# =========================
def build_model(input_dim, lr):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(224, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="out"),
        ]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# =========================
# GRID SEARCH: learning_rate x epochs
# =========================
LR_CANDIDATES = [0.25, 0.1, 0.05, 0.01, 0.005]
EPOCHS_CANDIDATES = [5, 10, 15, 20]
BATCH_SIZE = 256

results = []
# best = {"val_acc": -np.inf, "lr": None, "epochs": None}
best = {"recall_1": -np.inf, "lr": None, "epochs": None}


for lr in LR_CANDIDATES:
    for ep in EPOCHS_CANDIDATES:
        model = build_model(X_train.shape[1], lr)

        hist = model.fit(
            X_train,
            y_train,
            epochs=ep,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        # --- NUEVO: seleccionar por recall de la clase 1 en validación ---
        y_val_scores = model.predict(X_test, verbose=0).ravel()
        y_val_pred = (y_val_scores >= 0.2).astype(int)  # mismo umbral que usas
        rec1 = recall_score(y_test, y_val_pred, pos_label=1)

        # (opcional: guarda también val_acc para referencia)
        val_acc = float(hist.history["val_accuracy"][-1])
        train_acc = float(hist.history["accuracy"][-1])

        results.append((lr, ep, train_acc, val_acc, rec1))

        if rec1 > best["recall_1"]:
            best.update({"recall_1": rec1, "lr": lr, "epochs": ep})

        """
        # Tomamos la última val_accuracy (puedes usar max(hist.history["val_accuracy"]) si prefieres)
        val_acc = float(hist.history["val_accuracy"][-1])
        train_acc = float(hist.history["accuracy"][-1])

        results.append((lr, ep, train_acc, val_acc))
        if val_acc > best["val_acc"]:
            best.update({"val_acc": val_acc, "lr": lr, "epochs": ep})
            
        """


"""
# Mostrar resultados ordenados por val_accuracy desc
results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
print("\n=== Grid Search Summary (sorted by val_accuracy) ===")
print("lr\t\tepochs\ttrain_acc\tval_acc")
for lr, ep, tr, va in results_sorted:
    print(f"{lr:.5f}\t{ep:>3}\t{tr:.4f}\t\t{va:.4f}")

print(
    f"\n>> Mejor combinación: lr={best['lr']} | epochs={best['epochs']} | val_accuracy={best['val_acc']:.4f}"
)
"""


# Mostrar resultados ordenados por recall de la clase 1 (desc)
results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
print("\n=== Grid Search Summary (sorted by recall_1) ===")
print("lr\t\tepochs\ttrain_acc\tval_acc\trecall_1")
for lr, ep, tr, va, r1 in results_sorted:
    print(f"{lr:.5f}\t{ep:>3}\t{tr:.4f}\t\t{va:.4f}\t\t{r1:.4f}")

print(
    f"\n>> Mejor combinación: lr={best['lr']} | epochs={best['epochs']} | recall_1={best['recall_1']:.4f}"
)
# =========================
# Re-entrenar con la mejor combinación y evaluar
# =========================
UMBRAL = 0.2  # tu umbral de decisión

best_model = build_model(X_train.shape[1], best["lr"])
best_model.fit(
    X_train,
    y_train,
    epochs=best["epochs"],
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=0,
)

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# Predicciones y métricas con umbral 0.2 (como en tu base)
y_pred = best_model.predict(X_test, verbose=0).ravel()
y_pred_classes = (y_pred >= UMBRAL).astype(int)

y_true_classes = y_test.values if isinstance(y_test, pd.Series) else y_test
print("\nClassification Report (threshold=0.2):")
print(classification_report(y_true_classes, y_pred_classes, digits=4))

cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:\n", cm)

"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
