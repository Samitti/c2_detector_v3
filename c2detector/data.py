from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd
from .config import (
    ACTIVE_IDLE_FEATURES, COLUMN_MAPPING, DUPLICATE_FEATURES, IDENTIFIER_COLUMNS,
    NONNEGATIVE_EXACT, NONNEGATIVE_KEYWORDS, STABLE_CORE_FEATURES, SUBFLOW_FEATURES,
    WINDOW_FEATURES,
)


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read CSV file {path}: {exc}") from exc
    if df.empty:
        raise ValueError(f"CSV file contains no rows: {path}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    return out.rename(columns=COLUMN_MAPPING)


def normalize_label(value: Any) -> Optional[int]:
    text = str(value).strip().lower()
    if text in {"0", "benign", "normal", "background", "legitimate"}: return 0
    if text in {"1", "bot", "botnet", "c2", "malicious", "malware"}: return 1
    if "bot" in text or "malware" in text or "c2" in text: return 1
    if "benign" in text or "normal" in text or "background" in text: return 0
    return None


def extract_labels(df: pd.DataFrame, *, source_name: str, all_malicious: bool = False):
    data = normalize_columns(df)
    label_col = next((c for c in ["Label", "label", "Class", "class"] if c in data.columns), None)
    if label_col is not None:
        mapped = data[label_col].map(normalize_label)
        known = mapped.notna()
        if known.any():
            return (
                data.loc[known].drop(columns=[label_col]).reset_index(drop=True),
                mapped.loc[known].astype(int).reset_index(drop=True),
                {"source": source_name, "label_method": f"mapped from {label_col}",
                 "unknown_rows_excluded": int((~known).sum()), "rows_used": int(known.sum())},
            )
    if all_malicious:
        if label_col is not None:
            data = data.drop(columns=[label_col])
        return (
            data.reset_index(drop=True),
            pd.Series(np.ones(len(data), dtype=int), name="binary_label"),
            {"source": source_name, "label_method": "explicit all-malicious assumption",
             "unknown_rows_excluded": 0, "rows_used": len(data)},
        )
    raise ValueError(
        f"No usable binary labels were found in {source_name}. "
        "For a verified botnet-only CTU capture, rerun with --ctu-all-malicious."
    )


def _is_nonnegative_feature(name: str) -> bool:
    return name in NONNEGATIVE_EXACT or any(keyword in name for keyword in NONNEGATIVE_KEYWORDS)


def apply_feature_profile(columns: Sequence[str], profile: str) -> list[str]:
    cols = [c for c in columns if c not in DUPLICATE_FEATURES]
    if profile == "all-cleaned":
        return cols
    if profile == "stable-core":
        selected = [c for c in cols if c in STABLE_CORE_FEATURES]
        return selected or cols
    if profile == "no-active-idle":
        return [c for c in cols if c not in ACTIVE_IDLE_FEATURES]
    if profile == "no-windows":
        return [c for c in cols if c not in WINDOW_FEATURES]
    if profile == "no-subflow":
        return [c for c in cols if c not in SUBFLOW_FEATURES]
    raise ValueError(f"Unknown feature profile: {profile}")


def numeric_features(
    df: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
    remove_constant: bool = True,
    clean_invalid: bool = True,
    feature_profile: str = "all-cleaned",
):
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
    converted = converted.drop(columns=[c for c in DUPLICATE_FEATURES if c in converted.columns], errors="ignore")
    array = converted.to_numpy(dtype=float, copy=False)
    non_finite = int((~np.isfinite(array)).sum())
    converted = converted.replace([np.inf, -np.inf], np.nan)

    invalid_counts: dict[str, int] = {}
    if clean_invalid:
        for col in converted.columns:
            if _is_nonnegative_feature(col):
                mask = converted[col] < 0
                count = int(mask.sum())
                if count:
                    invalid_counts[col] = count
                    converted.loc[mask, col] = np.nan
        if "Destination Port" in converted.columns:
            mask = (converted["Destination Port"] < 0) | (converted["Destination Port"] > 65535)
            count = int(mask.sum())
            if count:
                invalid_counts["Destination Port(out_of_range)"] = count
                converted.loc[mask, "Destination Port"] = np.nan

    missing = int(converted.isna().sum().sum())
    # Median imputation is less destructive than replacing all invalid values with zero.
    medians = converted.median(numeric_only=True).fillna(0.0)
    converted = converted.fillna(medians).fillna(0.0)

    selected = apply_feature_profile(converted.columns.tolist(), feature_profile)
    converted = converted.loc[:, selected]
    constants = [c for c in converted.columns if converted[c].nunique(dropna=False) <= 1]
    if remove_constant and constants:
        converted = converted.drop(columns=constants)

    quality_rows = [
        {"feature": name, "invalid_negative_or_range_values": count}
        for name, count in sorted(invalid_counts.items())
    ]
    return converted, identifiers, {
        "non_finite_values_replaced": non_finite,
        "missing_or_non_numeric_values_imputed": missing,
        "invalid_values_by_feature": invalid_counts,
        "invalid_values_total": int(sum(invalid_counts.values())),
        "imputation_method": "training/source median, then zero only when median unavailable",
        "feature_profile": feature_profile,
        "duplicate_features_removed": sorted(DUPLICATE_FEATURES.intersection(data.columns)),
        "constant_columns_detected": len(constants),
        "constant_columns_removed": len(constants) if remove_constant else 0,
        "constant_column_names": constants,
        "quality_rows": quality_rows,
    }


def align_features(df: pd.DataFrame, feature_names: Sequence[str]):
    data = df.copy()
    missing = [n for n in feature_names if n not in data.columns]
    extra = [n for n in data.columns if n not in feature_names]
    present = [n for n in feature_names if n in data.columns]
    for n in missing:
        data[n] = 0.0
    data = data.drop(columns=extra, errors="ignore").loc[:, list(feature_names)]
    coverage = len(present) / len(feature_names) if feature_names else 0.0
    return data, {
        "training_feature_count": len(feature_names),
        "source_feature_count": int(df.shape[1]),
        "matched_feature_count": len(present),
        "coverage_ratio": round(coverage, 6),
        "missing_features_filled_with_zero": missing,
        "extra_features_dropped": extra,
    }


def build_feature_alignment_report(source_df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    source_cols = set(source_df.columns); training_cols = set(feature_names); rows = []
    for name in feature_names:
        rows.append({"feature": name, "required_by_model": True, "present_in_source": name in source_cols,
                     "action": "used" if name in source_cols else "filled_with_zero"})
    for name in sorted(source_cols - training_cols):
        rows.append({"feature": name, "required_by_model": False, "present_in_source": True,
                     "action": "dropped_extra"})
    return pd.DataFrame(rows)


def build_distribution_report(training_df: pd.DataFrame, external_df: pd.DataFrame,
                              feature_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        train = pd.to_numeric(training_df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ext = pd.to_numeric(external_df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        t_med, e_med = float(train.median()), float(ext.median())
        q1, q3 = float(train.quantile(.25)), float(train.quantile(.75)); t_iqr = q3 - q1
        zero_gap = abs(float((ext == 0).mean()) - float((train == 0).mean()))
        if abs(t_iqr) < 1e-9:
            robust_shift = np.nan
            shift_type = "structural_zero_or_constant_baseline" if abs(e_med - t_med) > 0 else "none"
            warning = shift_type != "none" and zero_gap >= 0.20
        else:
            robust_shift = abs(e_med - t_med) / t_iqr
            shift_type = "median_shift"
            warning = robust_shift > 5.0 or zero_gap >= 0.50
        rows.append({
            "feature": name, "training_median": t_med, "external_median": e_med,
            "training_mean": float(train.mean()), "external_mean": float(ext.mean()),
            "training_zero_fraction": float((train == 0).mean()),
            "external_zero_fraction": float((ext == 0).mean()),
            "zero_fraction_gap": zero_gap,
            "training_iqr": t_iqr,
            "robust_median_shift_iqr": robust_shift,
            "shift_type": shift_type,
            "shift_warning": bool(warning),
        })
    report = pd.DataFrame(rows)
    report["sort_score"] = report["robust_median_shift_iqr"].fillna(0) + report["zero_fraction_gap"] * 10
    return report.sort_values("sort_score", ascending=False).drop(columns=["sort_score"])
