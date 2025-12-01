# ml/training/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)
from typing import Dict

def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    y_true: shape (N,) values 0/1
    y_pred_proba: shape (N,) probabilities in [0,1]
    """
    y_true = y_true.astype(int)
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {}
    try:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:
        metrics["accuracy"] = metrics["precision"] = metrics["recall"] = metrics["f1"] = 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")

    # FPR at high threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics["tpr"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return metrics
