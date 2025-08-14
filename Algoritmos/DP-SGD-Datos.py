import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import tensorflow_privacy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer


data = pd.read_csv("../Datos/2/Base.csv")

X = data.drop(columns=["fraud_bool"])
y = data["fraud_bool"]

# convert the categorical columns to dummies
X = pd.get_dummies(X, drop_first=True)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# convert the labels
y = tf.keras.utils.to_categorical(y, num_classes=2)

# TODO solve the imbalance problem

# Separate into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Define hyperparameters
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
# Define the model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(y_train.shape[1]),
    ]
)

# Define the loss function and the optimizer
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate,
)

loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction=tf.losses.Reduction.NONE
)
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

epsilon, _ = compute_dp_sgd_privacy(
    n=len(X_train),
    batch_size=batch_size,
    noise_multiplier=noise_multiplier,
    epochs=epochs,
    delta=1e-5,
)

print(f"Privacidad diferencial alcanzada: ε = {epsilon:.2f}, δ = 1e-5")


"""
This code was adapted from the TensorFlow Privacy tutorial on classification privacy.:

https://www.tensorflow.org/responsible_ai/privacy/tutorials/classification_privacy?hl=es-419

"""
