from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def print_classification_metrics(
    y_true,
    y_pred,
    y_prob: Optional[object] = None,
) -> None:
    """
    Print standard classification metrics:
    accuracy, precision, recall and optional ROC-AUC.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    """
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    if y_prob is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}")


def plot_roc_curve(
    y_true,
    y_prob,
    filename: str = "figures/roc_curve.png",
) -> None:
    """
    Plot ROC curve and save it as a PNG file.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    filename : str
        Where to save the ROC curve image.
    """
    figures_dir = Path(filename).parent
    figures_dir.mkdir(exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
