import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)


def evaluate_classification(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def get_classification_report(
    y_true,
    y_pred,
    target_names=None,
):
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
