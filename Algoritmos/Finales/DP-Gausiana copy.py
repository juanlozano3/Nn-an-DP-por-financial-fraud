import os
import time
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import estimator as tf_estimator

from tensorflow_privacy.privacy.optimizers import dp_optimizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample

import mlflow
import mlflow.tensorflow


GDP_REPO = "../Deep-Learning-with-GDP-Tensorflow"
sys.path.append(GDP_REPO)
from gdp_accountant import compute_epsP, compute_epsilon


# -----------------------------
# Modelo Keras (usado por Estimator)
# -----------------------------
def define_model(n_features: int) -> tf.keras.Model:
    """
    Define arquitectura MLP binaria.
    Recibe n_features explícitamente (Estimator no puede inferirlo aquí).
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def model(features, labels, mode, params):
    """
    model_fn para tf.Estimator con DP-SGD.
    - Construye el grafo (Keras)
    - Define pérdida per-ejemplo (vector) para DP
    - Métricas/Eval/Pred
    """
    # Extraer dimensión de entrada desde el tensor (estática si está disponible)
    x_tensor = features["x"]  # shape: [batch, n_features]
    n_features = x_tensor.shape[-1]
    # Fallback si la dimensión no es estática:
    if n_features is None:
        n_features = tf.shape(x_tensor)[-1]

    net = define_model(int(n_features))
    logits = net(x_tensor)  # probabilidades (sigmoid)

    # Predicciones (usar umbral 0.5 por defecto; si quieres 0.2, cambia aquí)
    class_ids = tf.cast(logits >= 0.5, tf.int64)
    predictions = {"class_ids": class_ids, "logits": logits}

    # --- PREDICT ---
    if mode == tf_estimator.ModeKeys.PREDICT:
        return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # A partir de aquí, TRAIN/EVAL necesitan labels bien formateadas
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 1))  # (B,1) para binaria

    # --- PÉRDIDA ---
    # Para DP, el optimizador espera pérdida por-ejemplo (vector) si num_microbatches>1
    loss_obj = tf.keras.losses.BinaryCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    vector_loss = loss_obj(y_true=labels, y_pred=logits)  # shape: (B,)
    scalar_loss = tf.reduce_mean(vector_loss)

    # --- MÉTRICAS ---
    # Alinear tipos/formas: accuracy compara int64 vs int64 y vectores 1-D
    labels_int = tf.cast(tf.squeeze(labels, axis=1), tf.int64)  # (B,)
    class_ids_1d = tf.squeeze(class_ids, axis=1)  # (B,)
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=labels_int, predictions=class_ids_1d
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
    # Importante: para DP, pasar vector_loss al minimize (per-ejemplo)
    train_op = optimizer.minimize(loss=vector_loss, global_step=global_step)

    # --- TRAIN ---
    if mode == tf_estimator.ModeKeys.TRAIN:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, train_op=train_op
        )

    # --- EVAL ---
    elif mode == tf_estimator.ModeKeys.EVAL:
        return tf_estimator.EstimatorSpec(
            mode=mode, loss=scalar_loss, eval_metric_ops={"accuracy": accuracy}
        )


def make_input_fn(X, y, batch_size, shuffle=True, repeat=True):
    """
    Envuelve numpy arrays en tf.data.Dataset para Estimator.
    """

    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(({"x": X}, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        if repeat:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        return ds

    return input_fn


def main():
    """
    Entrena y evalúa un clasificador con DP-SGD vía Estimator.
    Mantiene MLflow y el cómputo de privacidad (ε) usando tu GDP accountant.
    """
    mlflow.set_experiment("DP-Fraud-Detection")

    # -----------------------------
    # Carga y preparación de datos
    # -----------------------------
    data = pd.read_csv("../Datos/2/Base.csv")

    X_df = data.drop(columns=["fraud_bool"])
    y = data["fraud_bool"].astype(int).values

    # Separar por tipo
    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns

    # One-hot robusto a versiones de scikit-learn
    if len(categorical_cols) > 0:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            # scikit-learn < 1.2
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_df[categorical_cols])
    else:
        X_cat = np.empty((len(X_df), 0), dtype=np.float32)

    # Escalar numéricas
    scaler = StandardScaler()
    X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])
    X_num = X_df[numeric_cols].to_numpy(dtype=np.float32)

    # Concatenar numéricas + categóricas
    X = np.hstack([X_num, X_cat]).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Balanceo por downsample (manteniendo tu enfoque)
    # -----------------------------
    # Convertir a DataFrame/Series
    X_train_df = pd.DataFrame(X_train)
    y_train_s = pd.Series(y_train)

    # Dividir por clase
    X_majority = X_train_df[y_train_s == 0]
    X_minority = X_train_df[y_train_s == 1]
    y_majority = y_train_s[y_train_s == 0]
    y_minority = y_train_s[y_train_s == 1]

    # Downsample mayoría al tamaño de minoría
    X_majority_ds, y_majority_ds = resample(
        X_majority,
        y_majority,
        replace=False,
        n_samples=len(y_minority),
        random_state=42,
    )

    # Concatenar y barajar (para evitar sesgo de orden)
    X_train_bal = pd.concat([X_majority_ds, X_minority], axis=0)
    y_train_bal = pd.concat([y_majority_ds, y_minority], axis=0)

    # Shuffle
    shuffled = X_train_bal.assign(_y_=y_train_bal.values).sample(
        frac=1.0, random_state=42
    )
    y_train = shuffled.pop("_y_").to_numpy(dtype=np.int32)
    X_train = shuffled.to_numpy(dtype=np.float32)

    # Reporte tamaños
    print("X train:", len(X_train), "X test:", len(X_test))
    print("y train:", len(y_train), "y test:", len(y_test))
    print(
        "Distribución train balanceado:",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )

    # -----------------------------
    # Hiperparámetros
    # -----------------------------
    batch_size = 256
    total_epochs = 5
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)

    params = {
        "l2_norm_clip": 1.2,
        "noise_multiplier": 1.1,
        "num_microbatches": 32,  # si usas microbatches, vector_loss es obligatorio
        "learning_rate": 0.15,
        "unroll_microbatches": False,
    }

    # -----------------------------
    # Entrenamiento / Evaluación
    # -----------------------------
    with mlflow.start_run(run_name="DP-Gausiana"):
        # Log de hiperparámetros
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", total_epochs)
        mlflow.log_param("alg", "SGD")  # quitamos etiquetas irreales de PCA/VarSel

        # Estimator
        fraud_classifier = tf_estimator.Estimator(model_fn=model, params=params)

        for epoch in range(1, total_epochs + 1):
            t0 = time.time()

            # Train una época (steps controla cuántos batches)
            fraud_classifier.train(
                input_fn=make_input_fn(X_train, y_train, batch_size),
                steps=steps_per_epoch,
            )

            dt = time.time() - t0
            print(f"Epoch {epoch}/{total_epochs} - Time: {dt:.2f}s")

            # Eval
            eval_results = fraud_classifier.evaluate(
                input_fn=make_input_fn(
                    X_test, y_test, batch_size, shuffle=False, repeat=False
                ),
                steps=max(1, X_test.shape[0] // batch_size),
            )
            print(f"Evaluation: {eval_results}")

            # Predicciones para métricas detalladas
            predictions = list(
                fraud_classifier.predict(
                    input_fn=make_input_fn(
                        X_test, y_test, batch_size, shuffle=False, repeat=False
                    )
                )
            )

            # Métricas básicas
            mlflow.log_metric("eval_loss", float(eval_results["loss"]), step=epoch)
            if "accuracy" in eval_results:
                mlflow.log_metric(
                    "accuracy", float(eval_results["accuracy"]), step=epoch
                )

            # Umbral 0.5 (si prefieres 0.2, cambia aquí y en model_fn)
            y_pred = [int(p["logits"][0] >= 0.5) for p in predictions]
            y_true = y_test[: len(y_pred)]

            report = classification_report(y_true, y_pred, output_dict=True)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, digits=4))

            cm = confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            # Log métricas relevantes
            for cls in ["0", "1"]:
                mlflow.log_metric(
                    f"precision_class_{cls}", report[cls]["precision"], step=epoch
                )
                mlflow.log_metric(
                    f"recall_class_{cls}", report[cls]["recall"], step=epoch
                )
                mlflow.log_metric(
                    f"f1_class_{cls}", report[cls]["f1-score"], step=epoch
                )
            mlflow.log_metric(
                "precision_macro", report["macro avg"]["precision"], step=epoch
            )
            mlflow.log_metric("recall_macro", report["macro avg"]["recall"], step=epoch)
            mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"], step=epoch)
            mlflow.log_metric(
                "precision_weighted", report["weighted avg"]["precision"], step=epoch
            )
            mlflow.log_metric(
                "recall_weighted", report["weighted avg"]["recall"], step=epoch
            )
            mlflow.log_metric(
                "f1_weighted", report["weighted avg"]["f1-score"], step=epoch
            )

            # -----------------------------
            # Privacidad: ε con tu GDP accountant
            # -----------------------------
            if params["noise_multiplier"] > 0:
                epsilon = compute_epsP(
                    epoch,
                    params["noise_multiplier"],
                    X_train.shape[0],
                    batch_size,
                    1e-6,
                )
                mlflow.log_metric("epsilon", float(epsilon), step=epoch)
                print(f"DP-Gausiana Privacy after {epoch} epochs: ε = {epsilon:.2f}")


if __name__ == "__main__":
    main()
