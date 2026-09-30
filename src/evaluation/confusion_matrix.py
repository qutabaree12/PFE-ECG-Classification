import numpy as np
from sklearn.metrics import confusion_matrix


def normalized_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_sums = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
