"""import sklearn
import tensorflow as tf
import tensorflow_estimator as tf_estimator
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from sklearn.utils import resample
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import mlflow
import mlflow.tensorflow
import os
"""

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

GDP_REPO = "../Deep-Learning-with-GDP-Tensorflow"
sys.path.append(GDP_REPO)
from gdp_accountant import compute_epsP, compute_epsilon


def define_model(features):
    """
    Define the model architecture.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(features["x"].shape[1])),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def model(features, labels, mode, params):
    """
    Model Function for tf.Estimator.
    Define el modelo, la función de pérdida, el optimizador y las salidas según el modo.
    """
    # Construir el modelo
    model = define_model(features)

    # Forward pass: logits = salida sin activar
    logits = model(features["x"])

    # Definir predicciones (para modo PREDICT)
    predictions = {
        "class_ids": tf.cast(logits > 0.2, tf.int64),
        "logits": logits,
    }

    # --- MODO PREDICT ---
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)
        # Expandir labels para que tengan forma (batch_size, 1)
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))

    # --- FUNCIÓN DE PÉRDIDA ---
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    vector_loss = loss_fn(y_true=labels, y_pred=logits)

    scalar_loss = tf.reduce_mean(vector_loss)

    # --- MÉTRICAS ---
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions["class_ids"]
    )

    # --- OPTIMIZADOR DP ---
    optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
        l2_norm_clip=params["l2_norm_clip"],
        noise_multiplier=params["noise_multiplier"],
        num_microbatches=params["num_microbatches"],
        learning_rate=params["learning_rate"],
    )

    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_op = optimizer.minimize(loss=vector_loss, global_step=global_step)

    # --- MODO TRAIN ---
    if mode == tf_estimator.ModeKeys.TRAIN:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op
        )

    # --- MODO EVAL ---
    elif mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, eval_metric_ops={"accuracy": accuracy}
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
        dataset = dataset.batch(batch_size)
        return dataset

    return input_fn


def main():
    """
    Main function to run DP-SGD training with tf.Estimator.
    """
    mlflow.set_experiment("DP-Fraud-Detection")

    # Load and prepare dataset
    data = pd.read_csv("../Datos/2/Base.csv")

    # Split features and labels
    X = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Codificamos las categóricas
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical_cols])
    # transformar las numericas
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    # Combinamos con las numéricas
    X_num = X[numeric_cols].values
    X = np.hstack([X_num, X_cat])
    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
    sel.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convertir a numpy arrays y asegurarse de los tipos
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)
    unique_pre, counts_pre = np.unique(y_train, return_counts=True)
    print("Antes del balanceo:", dict(zip(unique_pre, counts_pre)))

    # Balanceo de clases en el set de entrenamiento

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

    # Training parameters
    batch_size = 256
    total_epochs = 5
    steps_per_epoch = X_train.shape[0] // batch_size

    # DP-SGD parameters
    params = {
        "l2_norm_clip": 1.2,
        "noise_multiplier": 1.1,
        "num_microbatches": 32,
        "learning_rate": 0.15,
    }

    with mlflow.start_run(run_name="DP-SGD-VarSelection"):
        # Log params
        mlflow.log_param("l2_norm_clip", params["l2_norm_clip"])
        mlflow.log_param("noise_multiplier", params["noise_multiplier"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("num_microbatches", params["num_microbatches"])
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", total_epochs)
        mlflow.log_param("alg", "SGD-VarSelection")

        # Entrenamiento y evaluación (tu bucle de epochs va aquí)

        # Estimator
        fraud_classifier = tf_estimator.Estimator(model_fn=model, params=params)

        # Training loop
        for epoch in range(1, total_epochs + 1):
            start_time = time.time()

            # Train
            fraud_classifier.train(
                input_fn=make_input_fn(X_train, y_train, batch_size),
                steps=steps_per_epoch,
            )

            end_time = time.time()
            print(f"Epoch {epoch}/{total_epochs} - Time: {end_time - start_time:.2f}s")

            # Evaluate
            eval_results = fraud_classifier.evaluate(
                input_fn=make_input_fn(
                    X_test, y_test, batch_size, shuffle=False, repeat=False
                ),
                steps=X_test.shape[0] // batch_size,
            )
            print(f"Evaluation: {eval_results}")

            # Classification report
            # Classification report
            predictions = list(
                fraud_classifier.predict(
                    input_fn=make_input_fn(
                        X_test, y_test, batch_size, shuffle=False, repeat=False
                    )
                )
            )

            mlflow.log_metric("accuracy", eval_results["accuracy"], step=epoch)

            y_pred = [1 if p["logits"][0] > 0.2 else 0 for p in predictions]

            report = classification_report(
                y_test[: len(y_pred)], y_pred, output_dict=True
            )

            # Construir un diccionario plano con todas las métricas
            metrics_dict = {
                "precision_class_0": report["0"]["precision"],
                "recall_class_0": report["0"]["recall"],
                "f1_class_0": report["0"]["f1-score"],
                "support_class_0": report["0"]["support"],
                "precision_class_1": report["1"]["precision"],
                "recall_class_1": report["1"]["recall"],
                "f1_class_1": report["1"]["f1-score"],
                "support_class_1": report["1"]["support"],
                "accuracy": report["accuracy"],
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "precision_weighted": report["weighted avg"]["precision"],
                "recall_weighted": report["weighted avg"]["recall"],
                "f1_weighted": report["weighted avg"]["f1-score"],
            }

            # Convertir a DataFrame (1 fila, múltiples columnas)
            metrics_df = pd.DataFrame([metrics_dict])

            # Guardar como CSV en el directorio actual
            metrics_file = f"classification_metrics_epoch_{epoch}.csv"
            metrics_df.to_csv(metrics_file, index=False)

            # Subir a MLflow
            mlflow.log_artifact(metrics_file)

            # Limpiar
            os.remove(metrics_file)

            print("\nClassification Report:")
            print(classification_report(y_test[: len(y_pred)], y_pred, digits=4))

            cm = confusion_matrix(y_test[: len(y_pred)], y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            # Convertir la matriz a DataFrame para que se vea bien
            cm_df = pd.DataFrame(
                cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
            )

            # Guardar como CSV
            cm_csv_path = f"confusion_matrix_epoch_{epoch}.csv"
            cm_df.to_csv(cm_csv_path)

            # Subir a MLflow
            mlflow.log_artifact(cm_csv_path)

            # (Opcional) Borrar local
            os.remove(cm_csv_path)

            #  privacy
            if params["noise_multiplier"] > 0:
                epsilon = compute_epsilon(
                    epoch,
                    params["noise_multiplier"],
                    X_train.shape[0],
                    batch_size,
                    1e-6,
                )

                mlflow.log_metric("epsilon", epsilon, step=epoch)

                print(f"DP-SGD Privacy after {epoch} epochs: ε = {epsilon:.2f}")


if __name__ == "__main__":
    main()
