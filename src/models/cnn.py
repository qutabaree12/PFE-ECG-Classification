import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn(input_shape=(360, 1), num_classes=5):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(
            48,
            3,
            activation="relu",
            kernel_regularizer=regularizers.l2(1.1875e-05),
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(
            64,
            5,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.0006214),
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Conv1D(
            64,
            3,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-05),
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(
            96,
            activation="relu",
            kernel_regularizer=regularizers.l2(1.1875e-05),
        ),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
