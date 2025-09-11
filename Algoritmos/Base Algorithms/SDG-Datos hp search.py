import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import resample
import tensorflow as tf
import tensorflow_privacy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
)
from tensorflow import keras
import keras_tuner as kt
from sklearn.metrics import classification_report

data = pd.read_csv("./Datos/2/Base.csv")

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

# Convert the categorical columns to dummies
X = pd.get_dummies(X, drop_first=True)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Resetear índices para que todo esté alineado
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

# 6. Imprimir las nuevas formas
print("X train: ", len(X_train))
print("X Test: ", len(X_test))
print("y Train: ", len(y_train))
print("y Test: ", len(y_test))
# Después del balanceo
unique_post, counts_post = np.unique(y_train, return_counts=True)
print("Después del balanceo:", dict(zip(unique_post, counts_post)))

# Define hyperparameters
epochs = 10
batch_size = 256
learning_rate = 0.25

# Define the model


def model_builder(hp):
    model = keras.Sequential()

    # Primera capa densa con unidades variables
    hp_units1 = hp.Int("units1", min_value=32, max_value=512, step=32)
    model.add(
        keras.layers.Dense(
            units=hp_units1, activation="relu", input_shape=(X_train.shape[1],)
        )
    )

    # Segunda capa densa con unidades variables
    hp_units2 = hp.Int("units2", min_value=32, max_value=256, step=32)
    model.add(keras.layers.Dense(units=hp_units2, activation="relu"))

    # Capa de salida (sigmoid si es multiclase one-hot)
    #Cambio MLmodel.add(keras.layers.Dense(y_train.shape[1], activation="sigmoid"))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    # Learning rate del optimizador
    hp_learning_rate = hp.Choice("learning_rate", values=[0.01, 0.05, 0.1, 0.25])

    optimizer = keras.optimizers.SGD(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        #Cambio ML loss=keras.losses.CategoricalCrossentropy(from_logits=False),
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
stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Realiza la búsqueda de hiperparámetros
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(
    f"""
La búsqueda de hiperparámetros ha finalizado.
Mejor número de unidades en la primera capa densa: {best_hps.get('units1')}
Mejor número de unidades en la segunda capa densa: {best_hps.get('units2')}
Mejor learning rate: {best_hps.get('learning_rate')}
"""
)


# Define the loss function and the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Train the model
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# ==== NUEVO BLOQUE: Classification report ====

# Obtener predicciones
y_pred = model.predict(X_test)

# Convertir de one-hot a etiquetas
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Imprimir classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, digits=4))


"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
