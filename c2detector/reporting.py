from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from .utils import ensure_parent


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def save_validation_artifacts(y_true, y_pred, y_prob, metrics: Dict[str, Any], metrics_csv: Path,
                              confusion_plot: Path, roc_plot: Path):
    rows = [
        ("accuracy", metrics.get("accuracy")),
        ("roc_auc", metrics.get("roc_auc")),
        ("false_positive_rate", metrics.get("false_positive_rate")),
        ("false_negative_rate", metrics.get("false_negative_rate")),
        ("precision_c2", metrics["classification_report"]["c2_malicious"]["precision"]),
        ("recall_c2", metrics["classification_report"]["c2_malicious"]["recall"]),
        ("f1_c2", metrics["classification_report"]["c2_malicious"]["f1-score"]),
    ]
    ensure_parent(metrics_csv)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(metrics_csv, index=False)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ensure_parent(confusion_plot)
    plt.figure(figsize=(5, 4)); plt.imshow(cm)
    plt.xticks([0, 1], ["Benign", "C2"]); plt.yticks([0, 1], ["Benign", "C2"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2): plt.text(j, i, f"{cm[i,j]:,}", ha="center", va="center")
    plt.tight_layout(); plt.savefig(confusion_plot, dpi=160); plt.close()
    if pd.Series(y_true).nunique() == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ensure_parent(roc_plot)
        plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
        plt.tight_layout(); plt.savefig(roc_plot, dpi=160); plt.close()
