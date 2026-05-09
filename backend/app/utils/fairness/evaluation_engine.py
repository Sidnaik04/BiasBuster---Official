import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.utils.feature_encoder import encode_features_for_inference
from app.utils.prediction import predict_labels
from app.utils.sensitive_preprocessing import bin_age_column
from app.utils.target_encoder import encode_target_column, normalize_value
from app.utils.fairness.metrics import (
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    selection_rate,
    true_positive_rate,
)

logger = logging.getLogger(__name__)

# Common positive-class representations.
POSITIVE_TOKENS = {
    ">50k",
    ">50k.",
    "yes",
    "true",
    "positive",
    "1",
}


def _normalize_sensitive_columns(
    df: pd.DataFrame,
    sensitive_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    normalized = list(sensitive_columns)

    for idx, col in enumerate(list(normalized)):
        if col.lower() == "age":
            df = bin_age_column(df, col)
            normalized[idx] = f"{col}_group"

    return df, normalized


def _coerce_predictions(predictions: Any) -> np.ndarray:
    series = pd.Series(np.asarray(predictions).ravel())
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()

    # Fall back to stable categorical encoding.
    return pd.factorize(series.astype(str))[0].astype(int)


def _determine_positive_label(raw_target: pd.Series, encoded_target: pd.Series) -> int:
    raw_normalized = raw_target.dropna().map(normalize_value)
    raw_set = set(raw_normalized.astype(str).tolist())

    if raw_set.intersection(POSITIVE_TOKENS):
        # When encoded is binary, prefer class 1 for known positive tokens.
        unique_vals = sorted(encoded_target.dropna().unique().tolist())
        if len(unique_vals) == 2 and 1 in unique_vals:
            return 1

    unique_encoded = sorted(encoded_target.dropna().unique().tolist())
    if not unique_encoded:
        raise ValueError("Target column has no valid class values after encoding")

    # Deterministic ordering: highest encoded class is treated as positive.
    return int(unique_encoded[-1])


def _to_binary_labels(values: np.ndarray, positive_label: int) -> np.ndarray:
    arr = np.asarray(values).ravel()
    return (arr == positive_label).astype(int)


def _safe_roc_auc(model, X: pd.DataFrame, y_true_binary: np.ndarray) -> Optional[float]:
    try:
        candidate = model
        if isinstance(model, ThresholdOptimizer):
            candidate = getattr(model, "estimator_", getattr(model, "estimator", None))

        if candidate is None:
            return None

        if hasattr(candidate, "predict_proba"):
            scores = candidate.predict_proba(X)
            if hasattr(scores, "shape") and scores.shape[1] > 1:
                score_vec = np.asarray(scores[:, 1]).ravel()
            else:
                score_vec = np.asarray(scores).ravel()
        elif hasattr(candidate, "decision_function"):
            score_vec = np.asarray(candidate.decision_function(X)).ravel()
        else:
            return None

        if len(np.unique(y_true_binary)) < 2:
            return None

        return float(roc_auc_score(y_true_binary, score_vec))
    except Exception as exc:
        logger.warning("ROC-AUC unavailable: %s", str(exc))
        return None


def generate_predictions(
    model,
    X: pd.DataFrame,
    sensitive_features: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if X is None or X.empty:
        raise ValueError("Feature matrix is empty")

    try:
        raw = predict_labels(model, X, sensitive_features=sensitive_features)
    except Exception:
        encoded = encode_features_for_inference(X)
        raw = predict_labels(model, encoded, sensitive_features=sensitive_features)

    y_pred = _coerce_predictions(raw)
    if y_pred.size == 0:
        raise ValueError("Prediction vector is empty")
    if y_pred.shape[0] != X.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: predictions={y_pred.shape[0]}, rows={X.shape[0]}"
        )

    return {
        "predictions": y_pred,
        "distribution": pd.Series(y_pred).value_counts(dropna=False).to_dict(),
    }


def compute_fairness_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    sensitive_df: pd.DataFrame,
) -> Dict[str, Any]:
    if sensitive_df is None or sensitive_df.empty:
        raise ValueError("Sensitive feature frame is empty")
    if len(y_true_binary) != len(y_pred_binary) or len(y_true_binary) != len(
        sensitive_df
    ):
        raise ValueError("Input length mismatch for fairness metric computation")

    by_attribute: Dict[str, Dict[str, Any]] = {}
    aggregate_dpd = 0.0
    aggregate_eod = 0.0
    aggregate_dir = 1.0
    warnings: List[str] = []

    for col in sensitive_df.columns:
        sensitive_series = sensitive_df[col].astype(str)
        groups = sensitive_series.dropna().unique().tolist()
        if len(groups) < 2:
            raise ValueError(
                f"Sensitive column '{col}' must contain at least two groups, found {len(groups)}"
            )

        group_rates: Dict[str, float] = {}
        group_tprs: Dict[str, float] = {}

        for grp in groups:
            mask = sensitive_series == str(grp)
            y_true_group = y_true_binary[mask]
            y_pred_group = y_pred_binary[mask]
            if y_true_group.size == 0:
                continue

            group_rates[str(grp)] = selection_rate(y_pred_group)
            group_tprs[str(grp)] = true_positive_rate(y_true_group, y_pred_group)

        if len(group_rates) < 2:
            raise ValueError(
                f"Sensitive column '{col}' produced fewer than two non-empty groups"
            )

        dpd = float(demographic_parity_difference(group_rates))
        eod = float(equal_opportunity_difference(group_tprs))
        dir_ratio = float(disparate_impact_ratio(group_rates))

        by_attribute[col] = {
            "selection_rate": group_rates,
            "true_positive_rate": group_tprs,
            "dpd": dpd,
            "eod": eod,
            "dir": dir_ratio,
        }

        aggregate_dpd = max(aggregate_dpd, abs(dpd))
        aggregate_eod = max(aggregate_eod, abs(eod))
        aggregate_dir = min(aggregate_dir, dir_ratio)

    fairness_score = float(np.clip(1 - ((aggregate_dpd + aggregate_eod) / 2), 0.0, 1.0))

    return {
        "aggregate": {
            "dpd": float(aggregate_dpd),
            "eod": float(aggregate_eod),
            "dir": float(aggregate_dir),
            "fairness_score": fairness_score,
        },
        "by_attribute": by_attribute,
        "warnings": warnings,
    }


def compute_performance_metrics(
    model,
    X: pd.DataFrame,
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
) -> Dict[str, Any]:
    if len(y_true_binary) != len(y_pred_binary):
        raise ValueError("Input length mismatch for performance metric computation")

    unique_labels = np.unique(y_true_binary)
    if unique_labels.size < 2:
        raise ValueError("Ground-truth labels must contain at least two classes")

    metrics = {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
    }

    warnings: List[str] = []
    roc_auc = _safe_roc_auc(model, X, y_true_binary)
    if roc_auc is None:
        warnings.append("ROC-AUC unavailable for current prediction/score distribution")
    metrics["roc_auc"] = roc_auc

    return {"metrics": metrics, "warnings": warnings}


def validate_metric_consistency(
    baseline_a: Dict[str, float],
    baseline_b: Dict[str, float],
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    compared_keys = ("dpd", "eod", "dir")
    deltas = {}
    warnings = []

    for key in compared_keys:
        if key not in baseline_a or key not in baseline_b:
            continue
        delta = abs(float(baseline_a[key]) - float(baseline_b[key]))
        deltas[key] = delta
        if delta > tolerance:
            warnings.append(
                f"Metric inconsistency detected for {key}: delta={delta:.6f} exceeds tolerance={tolerance:.6f}"
            )

    return {
        "consistent": len(warnings) == 0,
        "tolerance": tolerance,
        "deltas": deltas,
        "warnings": warnings,
    }


def evaluate_model_fairness(
    df: pd.DataFrame,
    model,
    target_column: str,
    sensitive_columns: List[str],
) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Dataset is empty or invalid")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    raw_target = df[target_column].copy()

    prepared_df, target_info = encode_target_column(df.copy(), target_column)
    prepared_df, normalized_sensitive = _normalize_sensitive_columns(
        prepared_df,
        sensitive_columns,
    )

    for col in normalized_sensitive:
        if col not in prepared_df.columns:
            raise ValueError(f"Sensitive column '{col}' not found in dataset")
        prepared_df[col] = prepared_df[col].astype(str)

    y_encoded = prepared_df[target_column].astype(int).to_numpy()
    positive_label = _determine_positive_label(raw_target, prepared_df[target_column])
    y_true_binary = _to_binary_labels(y_encoded, positive_label)

    X = prepared_df.drop(columns=[target_column])

    sensitive_for_prediction = None
    if isinstance(model, ThresholdOptimizer):
        primary_sensitive = normalized_sensitive[0]
        sensitive_for_prediction = prepared_df[primary_sensitive]

    pred_result = generate_predictions(model, X, sensitive_for_prediction)
    y_pred_encoded = pred_result["predictions"]

    # Keep class mapping deterministic and identical to target mapping.
    y_pred_binary = _to_binary_labels(y_pred_encoded, positive_label)
    if len(np.unique(y_pred_binary)) < 2:
        logger.warning(
            "Predictions contain a single class; fairness may still compute but ROC-AUC may be unavailable"
        )

    sensitive_df = prepared_df[normalized_sensitive]

    fairness_result = compute_fairness_metrics(
        y_true_binary, y_pred_binary, sensitive_df
    )
    performance_result = compute_performance_metrics(
        model,
        X,
        y_true_binary,
        y_pred_binary,
    )

    diagnostics = {
        "dataset_shape": prepared_df.shape,
        "target_distribution": pd.Series(y_true_binary)
        .value_counts(dropna=False)
        .to_dict(),
        "prediction_distribution": pred_result["distribution"],
        "sensitive_groups": {
            col: prepared_df[col].value_counts(dropna=False).to_dict()
            for col in normalized_sensitive
        },
        "positive_label_mapping": {
            "positive_label": int(positive_label),
            "audit_mode": target_info.get("audit_mode"),
        },
    }

    warnings = []
    warnings.extend(fairness_result.get("warnings", []))
    warnings.extend(performance_result.get("warnings", []))

    logger.info(
        "[evaluation_engine] shape=%s pred_dist=%s target_dist=%s positive_label=%s",
        diagnostics["dataset_shape"],
        diagnostics["prediction_distribution"],
        diagnostics["target_distribution"],
        positive_label,
    )

    return {
        "target_info": target_info,
        "sensitive_columns": normalized_sensitive,
        "fairness": fairness_result,
        "performance": performance_result["metrics"],
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
