import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.cnn import build_cnn


def create_callbacks(
    patience=5,
    min_lr=1e-5,
):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=patience,
            min_lr=min_lr,
        ),
    ]


def train_cnn(
    x_train,
    y_train,
    x_val,
    y_val,
    class_weight=None,
    epochs=40,
    batch_size=32,
    model_path=None,
):
    x_train = np.asarray(x_train)
    x_val = np.asarray(x_val)

    if x_train.ndim == 2:
        x_train = x_train.reshape((-1, x_train.shape[1], 1))

    if x_val.ndim == 2:
        x_val = x_val.reshape((-1, x_val.shape[1], 1))

    model = build_cnn(
        input_shape=x_train.shape[1:],
        num_classes=len(np.unique(y_train)),
    )

    callbacks = create_callbacks()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)

    return model, history
