#!/usr/bin/env python3
"""
C2 Traffic Detector v2.1

Training mode:
    python3 c2_detector_v2_1.py train

Prediction mode:
    python3 c2_detector_v2_1.py predict --input new_flows.csv

Direct PCAP prediction:
    python3 c2_detector_v2_1.py predict-pcap --input suspect.pcap

Synthetic smoke test:
    python3 c2_detector_v2_1.py self-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import logging
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


APP_NAME = "C2 Traffic Detector"
APP_VERSION = "2.1"
RANDOM_STATE = 42
DEFAULT_CICIDS_DIR = Path("data/cicids2017/MachineLearningCVE")
DEFAULT_BENIGN_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
DEFAULT_BOTNET_FILE = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
DEFAULT_CTU_FILE = Path("data/ctu13/ctu13_flows/botnet-capture-20110810-neris.pcap_Flow.csv")
OUTPUT_ROOT = Path("output")
DEFAULT_MODEL = OUTPUT_ROOT / "models/c2_model.json"
DEFAULT_TRAINING_REPORT = OUTPUT_ROOT / "reports/training_report.json"
DEFAULT_SHAP_PLOT = OUTPUT_ROOT / "shap/shap_summary.png"
DEFAULT_PREDICTION_REPORT = OUTPUT_ROOT / "reports/forensic_report.json"
DEFAULT_FLOW_DIR = OUTPUT_ROOT / "flows"
DEFAULT_METRICS_CSV = OUTPUT_ROOT / "validation/metrics.csv"
DEFAULT_CONFUSION_PLOT = OUTPUT_ROOT / "validation/confusion_matrix.png"
DEFAULT_ROC_PLOT = OUTPUT_ROOT / "validation/roc_curve.png"
DEFAULT_LOG = OUTPUT_ROOT / "logs/detector.log"

IDENTIFIER_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Timestamp",
    "Source IP", "Destination IP", "Protocol",
]

# CTU/CICFlowMeter naming differences. All target names are stripped.
COLUMN_MAPPING = {
    "Dst Port": "Destination Port",
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


def banner() -> None:
    print("\n" + "=" * 72)
    print(f"  {APP_NAME} v{APP_VERSION}")
    print("  CIS*6520 — University of Guelph")
    print("  Intended stakeholder context: RCMP NC3")
    print("=" * 72)


def step(message: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {message}")
    print("=" * 72)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)



def configure_logging(path: Path) -> None:
    ensure_parent(path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8")],
        force=True,
    )


def save_validation_artifacts(y_true, y_pred, y_prob, metrics: Dict[str, Any],
                              metrics_csv: Path, confusion_plot: Path, roc_plot: Path) -> None:
    ensure_parent(metrics_csv)
    rows = [
        ("accuracy", metrics.get("accuracy")),
        ("roc_auc", metrics.get("roc_auc")),
        ("false_positive_rate", metrics.get("false_positive_rate")),
        ("false_negative_rate", metrics.get("false_negative_rate")),
        ("precision_c2", metrics["classification_report"]["c2_malicious"]["precision"]),
        ("recall_c2", metrics["classification_report"]["c2_malicious"]["recall"]),
        ("f1_c2", metrics["classification_report"]["c2_malicious"]["f1-score"]),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(metrics_csv, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ensure_parent(confusion_plot)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.xticks([0, 1], ["Benign", "C2"])
    plt.yticks([0, 1], ["Benign", "C2"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(confusion_plot, dpi=160)
    plt.close()

    if pd.Series(y_true).nunique() == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ensure_parent(roc_plot)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(roc_plot, dpi=160)
        plt.close()


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV file {path}: {exc}") from exc


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = result.columns.astype(str).str.strip()
    result = result.rename(columns=COLUMN_MAPPING)
    return result


def normalize_label(value: Any) -> Optional[int]:
    """Map common dataset labels to 0=benign or 1=C2/botnet.

    Unknown labels return None so they are not silently mislabeled.
    """
    text = str(value).strip().lower()
    if text in {"0", "benign", "normal", "background", "legitimate"}:
        return 0
    if text in {"1", "bot", "botnet", "c2", "malicious", "malware"}:
        return 1
    if "bot" in text or "malware" in text or "c2" in text:
        return 1
    if "benign" in text or "normal" in text or "background" in text:
        return 0
    return None


def extract_labels(
    df: pd.DataFrame,
    *,
    source_name: str,
    all_malicious: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    data = normalize_columns(df)
    label_col = next((c for c in ["Label", "label", "Class", "class"] if c in data.columns), None)

    if label_col is not None:
        mapped = data[label_col].map(normalize_label)
        known = mapped.notna()
        unknown_count = int((~known).sum())
        if known.any():
            features = data.loc[known].drop(columns=[label_col]).reset_index(drop=True)
            labels = mapped.loc[known].astype(int).reset_index(drop=True)
            return features, labels, {
                "source": source_name,
                "label_method": f"mapped from {label_col}",
                "unknown_rows_excluded": unknown_count,
                "rows_used": int(known.sum()),
            }

    if all_malicious:
        if label_col is not None:
            data = data.drop(columns=[label_col])
        labels = pd.Series(np.ones(len(data), dtype=int), name="binary_label")
        return data.reset_index(drop=True), labels, {
            "source": source_name,
            "label_method": "explicit all-malicious assumption",
            "unknown_rows_excluded": 0,
            "rows_used": len(data),
        }

    raise ValueError(
        f"No usable binary labels were found in {source_name}. "
        "For a verified botnet-only CTU capture, rerun with --ctu-all-malicious."
    )


def numeric_features(
    df: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Return model features, preserved identifiers, and cleaning counts."""
    data = normalize_columns(df)
    identifiers = pd.DataFrame(index=data.index)

    if drop_identifiers:
        for column in IDENTIFIER_COLUMNS:
            if column in data.columns:
                identifiers[column] = data[column]
                data = data.drop(columns=[column])

    for col in ["Label", "label", "Class", "class", "binary_label"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    converted = data.apply(pd.to_numeric, errors="coerce")
    invalid_before = int((~np.isfinite(converted.to_numpy(dtype=float, copy=False))).sum())
    converted = converted.replace([np.inf, -np.inf], np.nan)
    missing_before = int(converted.isna().sum().sum())
    converted = converted.fillna(0.0)

    # Constant columns provide no information and can cause dataset-specific leakage.
    constant_cols = [c for c in converted.columns if converted[c].nunique(dropna=False) <= 1]
    if constant_cols:
        converted = converted.drop(columns=constant_cols)

    return converted, identifiers, {
        "non_finite_values_replaced": invalid_before,
        "missing_or_non_numeric_values_replaced": missing_before,
        "constant_columns_removed": len(constant_cols),
    }


def align_features(df: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    data = df.copy()
    missing = [name for name in feature_names if name not in data.columns]
    extra = [name for name in data.columns if name not in feature_names]
    for name in missing:
        data[name] = 0.0
    data = data.drop(columns=extra, errors="ignore")
    data = data.loc[:, list(feature_names)]
    return data, {"missing_features_filled_with_zero": missing, "extra_features_dropped": extra}


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["benign", "c2_malicious"],
        output_dict=True,
        zero_division=0,
    )
    roc_auc: Optional[float]
    if pd.Series(y_true).nunique() == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        roc_auc = None

    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "classification_report": report,
        "support": int(len(y_true)),
    }


def print_metrics(title: str, metrics: Dict[str, Any]) -> None:
    print(f"\n  {title}")
    print("  " + "-" * 66)
    report = metrics["classification_report"]
    print(f"  {'CLASS':<20}{'PRECISION':>12}{'RECALL':>10}{'F1':>10}{'SUPPORT':>12}")
    for key, display in [("benign", "Benign"), ("c2_malicious", "C2 Malicious")]:
        row = report[key]
        print(
            f"  {display:<20}{row['precision']:>12.4f}{row['recall']:>10.4f}"
            f"{row['f1-score']:>10.4f}{int(row['support']):>12,}"
        )
    print(f"\n  Accuracy             : {metrics['accuracy']:.6f}")
    print(f"  ROC-AUC              : {metrics['roc_auc']:.6f}" if metrics["roc_auc"] is not None else "  ROC-AUC              : N/A (one class only)")
    print(f"  False Positive Rate  : {metrics['false_positive_rate']:.8f}" if metrics["false_positive_rate"] is not None else "  False Positive Rate  : N/A (no benign examples)")
    print(f"  False Negative Rate  : {metrics['false_negative_rate']:.8f}" if metrics["false_negative_rate"] is not None else "  False Negative Rate  : N/A (no malicious examples)")
    cm = metrics["confusion_matrix"]
    print("\n  Confusion matrix")
    print(f"  Actual benign    -> predicted benign {cm['tn']:,}; predicted C2 {cm['fp']:,}")
    print(f"  Actual C2        -> predicted benign {cm['fn']:,}; predicted C2 {cm['tp']:,}")


def build_model(y_train: pd.Series) -> xgb.XGBClassifier:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    if negative == 0 or positive == 0:
        raise ValueError("Training data must contain both benign and malicious examples.")
    ratio = negative / positive
    print(f"  Training class ratio   : {ratio:.2f} benign per malicious")
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=ratio,
        random_state=RANDOM_STATE,
        n_jobs=1,
        tree_method="hist",
    )


def shap_matrix(explainer: shap.TreeExplainer, data: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(data)
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def create_global_shap_plot(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    output_path: Path,
    sample_size: int,
) -> List[str]:
    if len(X_test) == 0:
        return []
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = shap_matrix(explainer, sample)
    shap.summary_plot(values, sample, feature_names=sample.columns.tolist(), show=False)
    ensure_parent(output_path)
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close()
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    return [sample.columns[i] for i in order[:10]]


def external_validate(
    model: xgb.XGBClassifier,
    ctu_file: Optional[Path],
    feature_names: Sequence[str],
    *,
    ctu_all_malicious: bool,
) -> Optional[Dict[str, Any]]:
    if ctu_file is None or not ctu_file.exists():
        print("  CTU-13 validation      : skipped (file not found)")
        return None

    raw = read_csv_checked(ctu_file)
    ctu_features_raw, ctu_y, label_info = extract_labels(
        raw,
        source_name=str(ctu_file),
        all_malicious=ctu_all_malicious,
    )
    ctu_X, _, cleaning = numeric_features(ctu_features_raw)
    ctu_X, alignment = align_features(ctu_X, feature_names)
    predictions = model.predict(ctu_X)
    probabilities = model.predict_proba(ctu_X)[:, 1]
    metrics = compute_metrics(ctu_y, predictions, probabilities)
    metrics["labeling"] = label_info
    metrics["cleaning"] = cleaning
    metrics["feature_alignment"] = alignment
    metrics["dataset"] = str(ctu_file)
    return metrics


def deployment_assessment(internal: Dict[str, Any], external: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Research-oriented assessment, not an operational certification."""
    reasons: List[str] = []
    internal_c2 = internal["classification_report"]["c2_malicious"]
    if internal_c2["recall"] < 0.90:
        reasons.append("internal C2 recall is below 0.90")
    if internal_c2["precision"] < 0.90:
        reasons.append("internal C2 precision is below 0.90")
    fpr = internal.get("false_positive_rate")
    if fpr is None or fpr > 0.01:
        reasons.append("internal false-positive rate is unavailable or above 1%")
    if external is None:
        reasons.append("independent CTU-13 validation was not completed")
    else:
        external_c2 = external["classification_report"]["c2_malicious"]
        if external_c2["recall"] < 0.80:
            reasons.append("external CTU-13 C2 recall is below 0.80")
        if external["classification_report"]["benign"]["support"] == 0:
            reasons.append("CTU-13 validation contains no benign examples, so external FPR cannot be measured")

    return {
        "status": "research validation criteria met" if not reasons else "further validation required",
        "reasons": reasons,
        "warning": "This assessment is not RCMP approval, legal admissibility, or operational certification.",
    }


def train_command(args: argparse.Namespace) -> int:
    step("STEP 1 — Loading CICIDS2017 training data")
    benign_path = args.cicids_dir / args.benign_file
    botnet_path = args.cicids_dir / args.botnet_file
    benign_raw = read_csv_checked(benign_path)
    botnet_raw = read_csv_checked(botnet_path)

    benign_features, benign_y, benign_label_info = extract_labels(
        benign_raw,
        source_name=str(benign_path),
        all_malicious=False,
    )
    # Keep only genuinely benign Monday rows.
    benign_mask = benign_y == 0
    benign_features = benign_features.loc[benign_mask].reset_index(drop=True)
    benign_y = benign_y.loc[benign_mask].reset_index(drop=True)

    bot_features, bot_y, bot_label_info = extract_labels(
        botnet_raw,
        source_name=str(botnet_path),
        all_malicious=False,
    )
    # Keep only Bot/C2 rows from the Friday file.
    bot_mask = bot_y == 1
    bot_features = bot_features.loc[bot_mask].reset_index(drop=True)
    bot_y = bot_y.loc[bot_mask].reset_index(drop=True)

    if len(benign_features) == 0 or len(bot_features) == 0:
        raise ValueError("Could not find both benign and Bot-labeled CICIDS2017 rows.")

    combined_raw = pd.concat([benign_features, bot_features], ignore_index=True, sort=False)
    y = pd.concat([benign_y, bot_y], ignore_index=True)
    X, _, cleaning = numeric_features(combined_raw)

    print(f"  CICIDS benign rows     : {int((y == 0).sum()):,}")
    print(f"  CICIDS C2 rows         : {int((y == 1).sum()):,}")
    print(f"  Numeric features       : {X.shape[1]:,}")
    print(f"  Values replaced        : {cleaning['missing_or_non_numeric_values_replaced']:,}")

    step("STEP 2 — Creating independent CICIDS2017 train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Training rows          : {len(X_train):,}")
    print(f"  Internal test rows     : {len(X_test):,}")

    step("STEP 3 — Training XGBoost")
    model = build_model(y_train)
    model.fit(X_train, y_train)
    print("  Model trained successfully")

    step("STEP 4 — Internal CICIDS2017 evaluation")
    internal_pred = model.predict(X_test)
    internal_prob = model.predict_proba(X_test)[:, 1]
    internal_metrics = compute_metrics(y_test, internal_pred, internal_prob)
    print_metrics("CICIDS2017 held-out test set", internal_metrics)
    save_validation_artifacts(y_test, internal_pred, internal_prob, internal_metrics,
                              args.metrics_csv, args.confusion_plot, args.roc_plot)
    print(f"  Metrics CSV            : {args.metrics_csv}")
    print(f"  Confusion matrix plot  : {args.confusion_plot}")
    print(f"  ROC curve plot         : {args.roc_plot}")

    step("STEP 5 — Independent CTU-13 external validation")
    external_metrics = external_validate(
        model,
        args.ctu_file,
        X.columns.tolist(),
        ctu_all_malicious=args.ctu_all_malicious,
    )
    if external_metrics is not None:
        print_metrics("CTU-13 external validation", external_metrics)

    step("STEP 6 — Generating global SHAP summary")
    top_features = create_global_shap_plot(
        model,
        X_test,
        args.shap_output,
        args.shap_sample,
    )
    print(f"  SHAP plot              : {args.shap_output}")
    print("  Top features           : " + ", ".join(top_features[:6]))

    step("STEP 7 — Saving model and evidence-oriented report")
    ensure_parent(args.model_output)
    model.save_model(args.model_output)
    assessment = deployment_assessment(internal_metrics, external_metrics)

    report = {
        "report_metadata": {
            "generated_at_utc": utc_now(),
            "tool": f"{APP_NAME} v{APP_VERSION}",
            "purpose": "Research evaluation of encrypted C2 flow detection",
            "model": "XGBoost binary classifier",
            "training_dataset": "CICIDS2017",
            "external_validation_dataset": "CTU-13" if external_metrics else None,
            "stakeholder_context": "RCMP NC3",
        },
        "data_provenance": {
            "cicids_benign_file": str(benign_path),
            "cicids_botnet_file": str(botnet_path),
            "ctu13_file": str(args.ctu_file) if args.ctu_file else None,
            "benign_labeling": benign_label_info,
            "botnet_labeling": bot_label_info,
            "feature_count": len(X.columns),
            "feature_names": X.columns.tolist(),
            "cleaning": cleaning,
        },
        "internal_cicids2017_evaluation": internal_metrics,
        "external_ctu13_evaluation": external_metrics,
        "global_explainability": {
            "method": "SHAP TreeExplainer",
            "sample_size": min(args.shap_sample, len(X_test)),
            "plot": str(args.shap_output),
            "top_features": top_features,
            "note": "This is a global summary. Prediction mode generates a separate explanation for each alert.",
        },
        "research_readiness_assessment": assessment,
        "output_files": {
            "model": str(args.model_output),
            "training_report": str(args.report_output),
            "shap_plot": str(args.shap_output),
            "metrics_csv": str(args.metrics_csv),
            "confusion_matrix_plot": str(args.confusion_plot),
            "roc_curve_plot": str(args.roc_plot),
            "log": str(args.log_output),
        },
    }
    ensure_parent(args.report_output)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"  Model saved            : {args.model_output}")
    print(f"  Report saved           : {args.report_output}")
    print(f"  Assessment             : {assessment['status']}")
    return 0


def top_shap_reasons(
    feature_names: Sequence[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    top_n: int,
) -> List[Dict[str, Any]]:
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    reasons = []
    for idx in order:
        contribution = float(shap_values[idx])
        reasons.append({
            "feature": str(feature_names[idx]),
            "value": float(feature_values[idx]),
            "shap_contribution": contribution,
            "effect": "toward C2" if contribution > 0 else "toward benign",
        })
    return reasons


def predict_command(args: argparse.Namespace) -> int:
    if not args.model.exists():
        raise FileNotFoundError(f"Saved model not found: {args.model}")

    step("PREDICT 1 — Loading model and traffic CSV")
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    feature_names = model.get_booster().feature_names
    if not feature_names:
        raise RuntimeError("The saved model does not contain feature names.")

    raw = read_csv_checked(args.input)
    normalized = normalize_columns(raw)
    X_raw, identifiers, cleaning = numeric_features(normalized)
    X, alignment = align_features(X_raw, feature_names)
    print(f"  Flows loaded           : {len(X):,}")
    print(f"  Model features         : {len(feature_names):,}")

    step("PREDICT 2 — Detecting C2 flows")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    alert_indices = np.flatnonzero(predictions == 1)
    print(f"  C2 alerts              : {len(alert_indices):,}")

    step("PREDICT 3 — Generating per-alert SHAP explanations")
    alerts: List[Dict[str, Any]] = []
    if len(alert_indices):
        alert_X = X.iloc[alert_indices]
        explainer = shap.TreeExplainer(model)
        alert_shap = shap_matrix(explainer, alert_X)

        for position, row_index in enumerate(alert_indices):
            identity: Dict[str, Any] = {}
            if not identifiers.empty:
                for key, value in identifiers.iloc[row_index].dropna().to_dict().items():
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    identity[key] = value

            alerts.append({
                "event_id": f"c2-{int(row_index):08d}",
                "source_row": int(row_index),
                "event_type": "encrypted_c2_suspected",
                "severity": "high" if probabilities[row_index] >= 0.90 else "medium",
                "verdict": "C2_MALICIOUS",
                "confidence": round(float(probabilities[row_index]), 6),
                "network_identifiers": identity,
                "explanation_method": "SHAP TreeExplainer",
                "top_reasons": top_shap_reasons(
                    feature_names,
                    X.iloc[row_index].to_numpy(dtype=float),
                    alert_shap[position],
                    args.top_reasons,
                ),
            })

    print(f"  Explained alerts       : {len(alerts):,}")

    report = {
        "report_metadata": {
            "generated_at_utc": utc_now(),
            "tool": f"{APP_NAME} v{APP_VERSION}",
            "input_file": str(args.input),
            "source_pcap": str(getattr(args, "source_pcap", "")) or None,
            "flow_extractor": getattr(args, "flow_extractor", None),
            "model_used": str(args.model),
            "schema": "generic SIEM-oriented JSON; validate field mapping before production ingestion",
            "stakeholder_context": "RCMP NC3",
        },
        "summary": {
            "total_flows_analyzed": len(X),
            "c2_alerts_raised": len(alerts),
            "alert_rate": round(len(alerts) / len(X), 8) if len(X) else 0.0,
        },
        "data_quality": {
            "cleaning": cleaning,
            "feature_alignment": alignment,
        },
        "alerts": alerts,
        "limitations": [
            "An alert is an investigative lead, not proof of criminal activity.",
            "SHAP explains model influence; it does not establish legal admissibility.",
            "SIEM field mapping must be tested against the target platform before deployment.",
        ],
    }

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"  Report saved           : {args.output}")
    return 0



def run_cicflowmeter(
    pcap_path: Path,
    csv_output: Path,
    executable: str,
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    """Convert a PCAP/PCAPNG file to CICFlowMeter CSV using its CLI."""
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    if pcap_path.suffix.lower() not in {".pcap", ".pcapng", ".cap"}:
        raise ValueError(
            f"Unsupported capture extension '{pcap_path.suffix}'. "
            "Use a .pcap, .pcapng, or .cap file."
        )

    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        raise RuntimeError(
            f"CICFlowMeter executable '{executable}' was not found in PATH. "
            "Install the Python CICFlowMeter CLI, then verify with 'cicflowmeter --help', "
            "or provide its path with --cicflowmeter-bin."
        )

    ensure_parent(csv_output)
    if csv_output.exists():
        if not overwrite:
            raise ValueError(
                f"Flow CSV already exists: {csv_output}. "
                "Use --overwrite-flow-csv to replace it."
            )
        csv_output.unlink()

    command = [resolved_executable, "-f", str(pcap_path), "-c", str(csv_output)]
    print("  Flow extractor command : " + " ".join(command))
    logging.info("Running flow extractor: %s", command)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not start CICFlowMeter: {exc}") from exc

    if completed.stdout.strip():
        logging.info("CICFlowMeter stdout: %s", completed.stdout.strip())
    if completed.stderr.strip():
        logging.info("CICFlowMeter stderr: %s", completed.stderr.strip())

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(
            f"CICFlowMeter failed with exit code {completed.returncode}: {detail}"
        )
    if not csv_output.exists() or csv_output.stat().st_size == 0:
        raise RuntimeError(
            f"CICFlowMeter completed but did not create a usable CSV: {csv_output}"
        )

    # Validate early so the user gets a clear extraction error before model prediction.
    extracted = read_csv_checked(csv_output)
    if extracted.empty:
        raise RuntimeError(
            "CICFlowMeter produced an empty CSV. The capture may contain no supported IP flows."
        )

    return {
        "name": "CICFlowMeter CLI",
        "executable": resolved_executable,
        "command": command,
        "return_code": completed.returncode,
        "flow_csv": str(csv_output),
        "flows_extracted": len(extracted),
    }


def predict_pcap_command(args: argparse.Namespace) -> int:
    """Extract flow features from a capture and run the existing prediction pipeline."""
    step("PCAP 1 — Converting packet capture to CICFlowMeter CSV")
    pcap_path = args.input.resolve()
    csv_output = args.flow_csv
    if csv_output is None:
        csv_output = DEFAULT_FLOW_DIR / f"{pcap_path.stem}_Flow.csv"

    extractor_info = run_cicflowmeter(
        pcap_path,
        csv_output,
        args.cicflowmeter_bin,
        overwrite=args.overwrite_flow_csv,
    )
    print(f"  Flows extracted        : {extractor_info['flows_extracted']:,}")
    print(f"  Flow CSV saved         : {csv_output}")

    report_output = args.output
    if report_output is None:
        report_output = OUTPUT_ROOT / "reports" / f"{pcap_path.stem}_forensic_report.json"

    predict_args = argparse.Namespace(
        input=csv_output,
        model=args.model,
        output=report_output,
        log_output=args.log_output,
        top_reasons=args.top_reasons,
        source_pcap=pcap_path,
        flow_extractor=extractor_info,
    )
    return predict_command(predict_args)

def self_test_command(args: argparse.Namespace) -> int:
    """Fast end-to-end test using synthetic CICIDS-like and CTU-like data."""
    step("SELF-TEST — Creating synthetic datasets")
    rng = np.random.default_rng(RANDOM_STATE)
    work = args.workdir
    cicids = work / "data/cicids2017/MachineLearningCVE"
    ctu_dir = work / "data/ctu13/ctu13_flows"
    cicids.mkdir(parents=True, exist_ok=True)
    ctu_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["Flow Duration", "Fwd Header Length", "Destination Port", "Flow IAT Min"]

    benign = pd.DataFrame({
        "Flow Duration": rng.normal(1000, 120, 240),
        "Fwd Header Length": rng.normal(32, 3, 240),
        "Destination Port": rng.choice([80, 443, 53], 240),
        "Flow IAT Min": rng.normal(120, 15, 240),
        "Label": "BENIGN",
    })
    bot = pd.DataFrame({
        "Flow Duration": rng.normal(120, 25, 90),
        "Fwd Header Length": rng.normal(60, 5, 90),
        "Destination Port": rng.choice([4444, 8081, 9001], 90),
        "Flow IAT Min": rng.normal(8, 2, 90),
        "Label": "Bot",
    })
    # Friday file can contain unrelated benign traffic; the trainer must select Bot only.
    friday = pd.concat([bot, benign.sample(30, random_state=RANDOM_STATE)], ignore_index=True)
    ctu_benign = benign.sample(50, random_state=1).copy()
    ctu_bot = bot.sample(50, random_state=2).copy()
    ctu = pd.concat([ctu_benign, ctu_bot], ignore_index=True)
    ctu = ctu.rename(columns={"Destination Port": "Dst Port"})

    benign.to_csv(cicids / DEFAULT_BENIGN_FILE, index=False)
    friday.to_csv(cicids / DEFAULT_BOTNET_FILE, index=False)
    ctu.to_csv(ctu_dir / DEFAULT_CTU_FILE.name, index=False)

    train_args = argparse.Namespace(
        cicids_dir=cicids,
        benign_file=DEFAULT_BENIGN_FILE,
        botnet_file=DEFAULT_BOTNET_FILE,
        ctu_file=ctu_dir / DEFAULT_CTU_FILE.name,
        ctu_all_malicious=False,
        test_size=0.2,
        shap_sample=50,
        model_output=work / "output/models/c2_model.json",
        report_output=work / "output/reports/training_report.json",
        shap_output=work / "output/shap/shap_summary.png",
        metrics_csv=work / "output/validation/metrics.csv",
        confusion_plot=work / "output/validation/confusion_matrix.png",
        roc_plot=work / "output/validation/roc_curve.png",
        log_output=work / "output/logs/detector.log",
    )
    train_command(train_args)

    prediction_input = ctu.copy()
    prediction_input.insert(0, "Flow ID", [f"flow-{i}" for i in range(len(prediction_input))])
    prediction_path = work / "new_flows.csv"
    prediction_input.to_csv(prediction_path, index=False)
    predict_args = argparse.Namespace(
        input=prediction_path,
        model=work / "output/models/c2_model.json",
        output=work / "output/reports/forensic_report.json",
        log_output=work / "output/logs/detector.log",
        top_reasons=3,
    )
    predict_command(predict_args)

    required = [
        work / "output/models/c2_model.json",
        work / "output/reports/training_report.json",
        work / "output/shap/shap_summary.png",
        work / "output/reports/forensic_report.json",
        work / "output/validation/metrics.csv",
        work / "output/validation/confusion_matrix.png",
        work / "output/validation/roc_curve.png",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Self-test did not create: {missing}")

    with (work / "output/reports/forensic_report.json").open(encoding="utf-8") as handle:
        prediction_report = json.load(handle)
    if prediction_report["summary"]["c2_alerts_raised"] == 0:
        raise RuntimeError("Self-test expected at least one C2 alert.")
    if any("top_reasons" not in alert for alert in prediction_report["alerts"]):
        raise RuntimeError("Per-alert SHAP explanations are missing.")

    print("\n  SELF-TEST PASSED")
    print(f"  Test artifacts         : {work}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train on CICIDS2017 and validate independently on CTU-13")
    train.add_argument("--cicids-dir", type=Path, default=DEFAULT_CICIDS_DIR)
    train.add_argument("--benign-file", default=DEFAULT_BENIGN_FILE)
    train.add_argument("--botnet-file", default=DEFAULT_BOTNET_FILE)
    train.add_argument("--ctu-file", type=Path, default=DEFAULT_CTU_FILE)
    train.add_argument(
        "--ctu-all-malicious",
        action="store_true",
        help="Use only when the CTU CSV is verified to contain botnet flows only and lacks usable labels",
    )
    train.add_argument("--test-size", type=float, default=0.20)
    train.add_argument("--shap-sample", type=int, default=500)
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--report-output", type=Path, default=DEFAULT_TRAINING_REPORT)
    train.add_argument("--shap-output", type=Path, default=DEFAULT_SHAP_PLOT)
    train.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    train.add_argument("--confusion-plot", type=Path, default=DEFAULT_CONFUSION_PLOT)
    train.add_argument("--roc-plot", type=Path, default=DEFAULT_ROC_PLOT)
    train.add_argument("--log-output", type=Path, default=DEFAULT_LOG)
    train.set_defaults(func=train_command)

    predict = sub.add_parser("predict", help="Analyze a new CICFlowMeter CSV with a saved model")
    predict.add_argument("--input", "-i", type=Path, required=True)
    predict.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--output", "-o", type=Path, default=DEFAULT_PREDICTION_REPORT)
    predict.add_argument("--top-reasons", type=int, default=5)
    predict.add_argument("--log-output", type=Path, default=DEFAULT_LOG)
    predict.set_defaults(func=predict_command)

    predict_pcap = sub.add_parser(
        "predict-pcap",
        help="Convert a PCAP/PCAPNG to CICFlowMeter CSV and analyze it with a saved model",
    )
    predict_pcap.add_argument("--input", "-i", type=Path, required=True, help="Input .pcap/.pcapng/.cap file")
    predict_pcap.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    predict_pcap.add_argument(
        "--output", "-o", type=Path, default=None,
        help="JSON report path; default: output/reports/<pcap-name>_forensic_report.json",
    )
    predict_pcap.add_argument(
        "--flow-csv", type=Path, default=None,
        help="Extracted flow CSV path; default: output/flows/<pcap-name>_Flow.csv",
    )
    predict_pcap.add_argument(
        "--cicflowmeter-bin", default="cicflowmeter",
        help="CICFlowMeter executable name or full path (default: cicflowmeter)",
    )
    predict_pcap.add_argument(
        "--overwrite-flow-csv", action="store_true",
        help="Replace an existing extracted flow CSV",
    )
    predict_pcap.add_argument("--top-reasons", type=int, default=5)
    predict_pcap.add_argument("--log-output", type=Path, default=DEFAULT_LOG)
    predict_pcap.set_defaults(func=predict_pcap_command)

    self_test = sub.add_parser("self-test", help="Run a complete synthetic smoke test")
    self_test.add_argument("--workdir", type=Path, default=Path("v21_self_test"))
    self_test.add_argument("--log-output", type=Path, default=Path("v21_self_test/output/logs/detector.log"))
    self_test.set_defaults(func=self_test_command)
    return parser


def main() -> int:
    banner()
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(getattr(args, "log_output", DEFAULT_LOG))
    logging.info("Started %s v%s command=%s", APP_NAME, APP_VERSION, args.command)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
