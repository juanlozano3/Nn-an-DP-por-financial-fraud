import tensorflow as tf
import numpy as np
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

print("Se importaron bien las librerías")


# Cargar y preprocesar el conjunto de datos

# Cargar el conjunto de datos MNIST
train, test = tf.keras.datasets.mnist.load_data()
# Los datos vienen como una tupla en donde se tiene los datos de train y los lables
train_data, train_labels = train
test_data, test_labels = test
# Se convierten las imágenes a un array para poderlo manejar en la red neuronal
# y se normalizan los datos
train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255
# Las imágenes son de 28x28, pero para la red neuronal se requiere que sean
# de 4 dimensiones
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

# Se convierten las etiquetas de entrenamiento en vectores one-hot
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Verificar que los datos están normalizados
assert train_data.min() == 0.0
assert train_data.max() == 1.0
assert test_data.min() == 0.0
assert test_data.max() == 1.0

# Definir hiperparámetros
epochs = 3
batch_size = 250

l2_norm_clip = 1.5
noise_multiplier = 1.3
num_microbatches = 250
learning_rate = 0.25

if batch_size % num_microbatches != 0:
    raise ValueError(
        "Batch size should be an integer multiple of the number of microbatches"
    )

# Construir el modelo
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, 8, strides=2, padding="same", activation="relu", input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(32, 4, strides=2, padding="valid", activation="relu"),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)


# Definir la función de pérdida y el optimizador
optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate,
)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE
)

# Entrenar el modelo
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(
    train_data,
    train_labels,
    epochs=epochs,
    validation_data=(test_data, test_labels),
    batch_size=batch_size,
)
