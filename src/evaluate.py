import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate(
    y_true,
    y_pred,
):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def print_report(
    y_true,
    y_pred,
    class_names,
):
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )