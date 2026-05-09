import os
import joblib
import logging
import asyncio
import uuid
import traceback
from types import SimpleNamespace
from typing import Dict, List, Any, Optional
from sqlalchemy import select
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from app.config import settings
from app.models.models import UploadRecord
from app.models.bias_audit import BiasAuditRecord
from app.models.bias_mitigation import BiasMitigationRun

from app.services.bias_service import run_bias_detection
from app.utils.core.dataset_loader import load_dataset
from app.utils.core.model_loader import load_model
from app.utils.core.preprocessing import preprocess_dataset
from app.utils.fairness.evaluator import evaluate_baseline
from app.utils.fairness.comparison import compare_metrics
from app.utils.fairness.evaluation_engine import (
    evaluate_model_fairness,
    validate_metric_consistency,
)

from app.utils.mitigation.smote import apply_smote
from app.utils.mitigation.reweighting import compute_sample_weights
from app.utils.mitigation.threshold import apply_threshold_optimizer
from app.utils.mitigation.recommender import recommend_strategy
from app.models.mitigation_ranking import MitigationRanking

from app.schemas.bias import BiasDetectRequest
from app.utils.target_encoder import encode_target_column
from app.utils.sensitive_preprocessing import bin_age_column

from sklearn.pipeline import Pipeline
from fairlearn.postprocessing import ThresholdOptimizer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Upload/Temporary directories (where files are initially uploaded)
# ------------------------------------------------------------------
TEMP_DATA_DIR = os.path.join(settings.TEMP_DIR, "datasets")
TEMP_MODEL_DIR = os.path.join(settings.TEMP_DIR, "models")

os.makedirs(TEMP_DATA_DIR, exist_ok=True)
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Artifact directories (where mitigated artifacts are stored)
# ------------------------------------------------------------------
ART_MODEL_DIR = os.path.join(settings.ARTIFACT_DIR, "models")
ART_DATA_DIR = os.path.join(settings.ARTIFACT_DIR, "datasets")

os.makedirs(ART_MODEL_DIR, exist_ok=True)
os.makedirs(ART_DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Fairness thresholds
# ------------------------------------------------------------------
FAIRNESS_THRESHOLDS = {
    "dpd": 0.10,  # Demographic Parity Difference
    "eod": 0.10,  # Equal Opportunity Difference
    "dir": 0.80,  # Disparate Impact Ratio
}

MITIGATION_STRATEGIES = ("threshold", "reweighting", "smote")


# ==================================================================
# VALIDATION AND STRATEGY FUNCTIONS
# ==================================================================


class MitigationPipelineError(Exception):
    def __init__(self, stage: str, message: str, trace_id: Optional[str] = None):
        self.stage = stage
        self.message = message
        self.trace_id = trace_id or str(uuid.uuid4())
        super().__init__(message)


def _failure_response(
    stage: str, message: str, trace_id: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error_stage": stage,
        "diagnostic_details": message,
        "trace_id": trace_id or str(uuid.uuid4()),
    }


def _validate_non_empty_frame(df: pd.DataFrame, stage: str, trace_id: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise MitigationPipelineError(
            stage, "Dataset failed to load as DataFrame", trace_id
        )
    if df.empty:
        raise MitigationPipelineError(stage, "Loaded dataset is empty", trace_id)


def _validate_prediction_vector(
    y_true,
    y_pred,
    stage: str,
    trace_id: str,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_pred_arr.size == 0:
        raise MitigationPipelineError(stage, "Prediction vector is empty", trace_id)
    if y_true_arr.size == 0:
        raise MitigationPipelineError(stage, "Ground-truth labels are empty", trace_id)
    if y_pred_arr.shape[0] != y_true_arr.shape[0]:
        raise MitigationPipelineError(
            stage,
            f"Prediction length mismatch: predictions={y_pred_arr.shape[0]}, labels={y_true_arr.shape[0]}",
            trace_id,
        )

    pred_unique = np.unique(y_pred_arr)
    label_unique = np.unique(y_true_arr)

    if pred_unique.size == 0:
        raise MitigationPipelineError(
            stage, "Predictions have no valid classes", trace_id
        )
    if label_unique.size < 2:
        raise MitigationPipelineError(
            stage,
            "Labels contain only one class; cannot evaluate fairness/performance reliably",
            trace_id,
        )
    warnings: List[str] = []
    if pred_unique.size < 2:
        warnings.append(
            "Predictions contain a single class; ROC-AUC may be unavailable"
        )

    return {
        "pred_unique": pred_unique.tolist(),
        "label_unique": label_unique.tolist(),
        "pred_distribution": pd.Series(y_pred_arr).value_counts(dropna=False).to_dict(),
        "label_distribution": pd.Series(y_true_arr)
        .value_counts(dropna=False)
        .to_dict(),
        "warnings": warnings,
    }


def _validate_sensitive_series(
    sensitive_series: pd.Series,
    sensitive_column: str,
    stage: str,
    trace_id: str,
) -> None:
    if sensitive_series is None:
        raise MitigationPipelineError(
            stage, f"Sensitive column '{sensitive_column}' is missing", trace_id
        )
    if sensitive_series.shape[0] == 0:
        raise MitigationPipelineError(
            stage, f"Sensitive column '{sensitive_column}' has no rows", trace_id
        )
    group_count = sensitive_series.dropna().nunique()
    if group_count < 2:
        raise MitigationPipelineError(
            stage,
            f"Sensitive column '{sensitive_column}' must contain at least two groups, found {group_count}",
            trace_id,
        )


def validate_mitigation_strategy(strategy_name: str) -> None:
    """
    Validate that the strategy_name is supported.

    Args:
        strategy_name: Name of the mitigation strategy

    Raises:
        ValueError: If strategy is not supported
    """
    if strategy_name not in MITIGATION_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy_name}'. "
            f"Supported strategies: {', '.join(MITIGATION_STRATEGIES)}"
        )


def _get_dataset_path(filename: str) -> str:
    """
    Get dataset path, checking TEMP_DIR first, then ARTIFACT_DIR.
    """
    # Try TEMP_DIR first (where uploads are stored)
    temp_path = os.path.join(TEMP_DATA_DIR, filename)
    if os.path.exists(temp_path):
        return temp_path

    # Fall back to ARTIFACT_DIR (for development/testing)
    artifact_path = os.path.join(ART_DATA_DIR, filename)
    if os.path.exists(artifact_path):
        return artifact_path

    # Return temp path as default (will fail with proper error message)
    return temp_path


def _get_model_path(filename: str) -> str:
    """
    Get model path, checking TEMP_DIR first, then ARTIFACT_DIR.
    """
    # Try TEMP_DIR first (where uploads are stored)
    temp_path = os.path.join(TEMP_MODEL_DIR, filename)
    if os.path.exists(temp_path):
        return temp_path

    # Fall back to ARTIFACT_DIR (for development/testing)
    artifact_path = os.path.join(ART_MODEL_DIR, filename)
    if os.path.exists(artifact_path):
        return artifact_path

    # Return temp path as default (will fail with proper error message)
    return temp_path


def _requires_mitigation(metrics: Dict[str, float]) -> bool:
    """Check if fairness metrics exceed thresholds."""
    dpd = abs(metrics.get("dpd", 0))
    eod = abs(metrics.get("eod", 0))
    dir_ratio = metrics.get("dir", 1.0)

    return (
        dpd > FAIRNESS_THRESHOLDS["dpd"]
        or eod > FAIRNESS_THRESHOLDS["eod"]
        or dir_ratio < FAIRNESS_THRESHOLDS["dir"]
    )


def _safe_roc_auc(model, X, y_true, sensitive_features=None):
    try:
        candidate = model
        if isinstance(model, ThresholdOptimizer):
            candidate = getattr(model, "estimator_", getattr(model, "estimator", None))

        if candidate is None:
            logger.warning("ROC-AUC skipped: no candidate estimator found")
            return None

        if hasattr(candidate, "predict_proba"):
            y_score = candidate.predict_proba(X)
            if hasattr(y_score, "shape") and y_score.shape[1] > 1:
                y_score = y_score[:, 1]
            else:
                y_score = np.asarray(y_score).ravel()
        elif hasattr(candidate, "decision_function"):
            y_score = candidate.decision_function(X)
        else:
            logger.warning(
                "ROC-AUC skipped: estimator '%s' has no predict_proba/decision_function",
                type(candidate).__name__,
            )
            return None

        y_true_arr = np.asarray(y_true)
        if len(np.unique(y_true_arr)) < 2:
            logger.warning("ROC-AUC skipped: y_true contains a single class")
            return None

        return float(roc_auc_score(y_true_arr, np.asarray(y_score).ravel()))
    except Exception as exc:
        logger.warning("ROC-AUC computation failed: %s", str(exc))
        return None


def _augment_performance_metrics(model, X, y_true, sensitive_features, metrics):
    augmented = dict(metrics)
    roc_auc = _safe_roc_auc(model, X, y_true, sensitive_features=sensitive_features)
    if roc_auc is not None and not np.isnan(roc_auc):
        augmented["roc_auc"] = float(roc_auc)
    else:
        augmented["roc_auc"] = None
    return augmented


def _compute_fairness_improvement(
    before_eval: Dict[str, Any], after_eval: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute fairness improvements from before to after mitigation using centralized metrics.
    """
    if not before_eval or not after_eval:
        raise ValueError("Cannot compute fairness improvement with empty evaluations")

    before_fairness = before_eval["fairness"]["aggregate"]
    after_fairness = after_eval["fairness"]["aggregate"]

    dpd_reduction = float(max(0.0, before_fairness["dpd"] - after_fairness["dpd"]))
    eod_reduction = float(max(0.0, before_fairness["eod"] - after_fairness["eod"]))
    fairness_score_gain = float(max(0.0, after_fairness["fairness_score"] - before_fairness["fairness_score"]))

    return {
        "dpd_reduction": dpd_reduction,
        "eod_reduction": eod_reduction,
        "fairness_score_gain": fairness_score_gain,
    }


def _compute_performance_impact(
    before_eval: Dict[str, Any], after_eval: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute performance metric changes from before to after.
    """
    if not before_eval or not after_eval:
        raise ValueError("Cannot compute performance impact with empty evaluations")

    before_perf = before_eval["performance"]
    after_perf = after_eval["performance"]

    accuracy_change = float(after_perf["accuracy"] - before_perf["accuracy"])
    f1_change = float(after_perf["f1"] - before_perf["f1"])

    return {
        "accuracy_change": accuracy_change,
        "f1_change": f1_change,
    }


def _generate_tradeoff_analysis(
    strategy_name: str,
    fairness_improvement: Dict[str, float],
    performance_impact: Dict[str, float],
    baseline_metrics: Dict[str, Any],
    mitigated_metrics: Dict[str, Any],
) -> str:
    """
    Generate human-readable tradeoff analysis.

    Args:
        strategy_name: Name of strategy applied
        fairness_improvement: Dict with fairness improvements
        performance_impact: Dict with performance changes
        baseline_metrics: Before metrics
        mitigated_metrics: After metrics

    Returns:
        Human-readable analysis string
    """
    fairness_gain = fairness_improvement.get("fairness_score_gain", 0.0)
    accuracy_change = performance_impact.get("accuracy_change", 0.0)
    dpd_reduction = fairness_improvement.get("dpd_reduction", 0.0)
    eod_reduction = fairness_improvement.get("eod_reduction", 0.0)

    if fairness_gain < 0.01:
        return "No significant fairness improvement detected. Review mitigation strategy effectiveness."

    if dpd_reduction > 0.15 and accuracy_change > -0.05:
        return f"{strategy_name.capitalize()} mitigation achieved strong fairness improvement with minimal performance impact."
    elif dpd_reduction > 0.15 and accuracy_change <= -0.05:
        return f"{strategy_name.capitalize()} mitigation significantly reduced demographic parity disparity but with notable accuracy trade-off."
    elif eod_reduction > 0.15 and accuracy_change > -0.05:
        return f"{strategy_name.capitalize()} mitigation improved equal opportunity with strong performance retention."
    elif accuracy_change < -0.10:
        return f"{strategy_name.capitalize()} mitigation trades off model accuracy for fairness gains."
    else:
        return f"{strategy_name.capitalize()} mitigation achieved balanced fairness-performance improvement."


async def run_bias_mitigation(payload, session):
    """
    Fully pipeline-aware bias mitigation execution engine.

    Pipeline Flow:
    1. Load dataset using upload_id
    2. Load uploaded model
    3. Preprocess dataset
    4. Generate baseline predictions
    5. Compute baseline fairness + performance metrics
    6. Apply selected mitigation strategy
    7. Generate mitigated predictions
    8. Compute updated metrics
    9. Compute fairness improvements and performance impacts
    10. Return comprehensive mitigation report

    Args:
        payload: BiasMitigationRequest with cleaned minimal input
        session: Database session

    Returns:
        Comprehensive mitigation report with before/after comparison
    """

    trace_id = str(uuid.uuid4())
    try:
        # ============================================
        # STEP 1: Load dataset and model from upload
        # ============================================
        record = (
            await session.execute(
                select(UploadRecord).where(UploadRecord.id == payload.upload_id)
            )
        ).scalar_one_or_none()

        if not record:
            raise ValueError(f"Upload record with id={payload.upload_id} not found")

        dataset_path = _get_dataset_path(record.dataset_filename)
        model_path = _get_model_path(record.model_filename)

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset file not found: {record.dataset_filename}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {record.model_filename}")

        df = load_dataset(dataset_path)
        model, preprocessor = load_model(model_path)
        _validate_non_empty_frame(df, "dataset_loading", trace_id)
        if model is None:
            raise MitigationPipelineError(
                "model_loading", "Loaded model is None", trace_id
            )
        if not hasattr(model, "predict"):
            raise MitigationPipelineError(
                "model_loading",
                f"Loaded model type '{type(model).__name__}' does not implement predict()",
                trace_id,
            )

        logger.info(
            "[mitigation:%s] Loaded dataset=%s shape=%s model=%s strategy=%s",
            trace_id,
            dataset_path,
            df.shape,
            type(model).__name__,
            payload.strategy_name,
        )

        # ============================================
        # STEP 2: Validate input
        # ============================================
        if payload.target_column not in df.columns:
            raise ValueError(
                f"Target column '{payload.target_column}' not found in dataset"
            )

        for col in payload.sensitive_columns:
            if col not in df.columns and col != "age_group":
                raise ValueError(f"Sensitive column '{col}' not found in dataset")

        # ============================================
        # STEP 3: Preprocess dataset + centralized baseline evaluation
        # ============================================
        baseline_eval = evaluate_model_fairness(
            df=df,
            model=model,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
        )

        sensitive_columns_eval = baseline_eval["sensitive_columns"]

        if not baseline_eval or "fairness" not in baseline_eval:
            raise MitigationPipelineError(
                "baseline_metric_computation",
                "No baseline metrics were computed for requested sensitive columns",
                trace_id,
            )

        logger.info(
            "[mitigation:%s] Baseline aggregate metrics=%s diagnostics=%s",
            trace_id,
            baseline_eval["fairness"]["aggregate"],
            baseline_eval["diagnostics"],
        )

        # Build encoded data used for training mitigation strategies.
        df_encoded, _ = encode_target_column(df, payload.target_column)
        _validate_non_empty_frame(df_encoded, "target_encoding", trace_id)
        y = df_encoded[payload.target_column]
        X_raw = df_encoded.drop(columns=[payload.target_column])

        if y.isna().all():
            raise MitigationPipelineError(
                "target_encoding",
                f"Encoded target column '{payload.target_column}' contains only NaN values",
                trace_id,
            )

        for col in payload.sensitive_columns:
            if col == "age_group" and "age" in X_raw.columns:
                X_raw = bin_age_column(X_raw, "age")

        # Split data for evaluation
        test_size = payload.strategy_config.get("test_size", 0.2)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info(
            "[mitigation:%s] Split complete train_shape=%s test_shape=%s y_train_dist=%s y_test_dist=%s",
            trace_id,
            X_train_raw.shape,
            X_test_raw.shape,
            pd.Series(y_train).value_counts(dropna=False).to_dict(),
            pd.Series(y_test).value_counts(dropna=False).to_dict(),
        )

        # ============================================
        # STEP 4 & 5: Baseline already computed by shared engine
        # ============================================

        # ============================================
        # STEP 6: Apply mitigation strategy
        # ============================================
        strategy_name = payload.strategy_name

        if strategy_name == "reweighting":
            model_mitigated = _apply_reweighting_for_mitigation(
                X_train_raw,
                y_train,
                model,
                payload.sensitive_columns,
                sensitive_columns_eval,
                baseline_eval,
            )
        elif strategy_name == "threshold":
            model_mitigated = _apply_threshold_for_mitigation(
                X_train_raw,
                y_train,
                model,
                payload.sensitive_columns,
                sensitive_columns_eval,
                baseline_eval,
                payload.strategy_config,
            )
        elif strategy_name == "smote":
            model_mitigated = _apply_smote_for_mitigation(
                X_train_raw,
                y_train,
                model,
                payload.sensitive_columns,
                sensitive_columns_eval,
                baseline_eval,
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        # ============================================
        # STEP 7-9: Centralized post-mitigation evaluation
        # ============================================
        after_eval = evaluate_model_fairness(
            df=df,
            model=model_mitigated,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
        )

        if not after_eval or "fairness" not in after_eval:
            raise MitigationPipelineError(
                "mitigated_metric_computation",
                "No mitigated metrics were computed for requested sensitive columns",
                trace_id,
            )

        logger.info(
            "[mitigation:%s] Post-mitigation aggregate metrics=%s diagnostics=%s",
            trace_id,
            after_eval["fairness"]["aggregate"],
            after_eval["diagnostics"],
        )

        # ============================================
        # STEP 10: Compute improvements and save artifacts
        # ============================================
        fairness_improvement = _compute_fairness_improvement(
            baseline_eval, after_eval
        )
        performance_impact = _compute_performance_impact(baseline_eval, after_eval)

        tradeoff_analysis = _generate_tradeoff_analysis(
            strategy_name,
            fairness_improvement,
            performance_impact,
            baseline_eval,
            after_eval,
        )

        # Save mitigated model
        artifact_model_path = os.path.join(
            ART_MODEL_DIR, f"model_{record.id}_{strategy_name}_mitigated.joblib"
        )
        joblib.dump(model_mitigated, artifact_model_path)

        # Save mitigated dataset if applicable
        artifact_dataset_path = None
        if strategy_name in ("reweighting", "smote"):
            mitigated_df = X_train_raw.copy()
            mitigated_df[payload.target_column] = y_train
            artifact_dataset_path = os.path.join(
                ART_DATA_DIR, f"dataset_{record.id}_{strategy_name}_mitigated.csv"
            )
            mitigated_df.to_csv(artifact_dataset_path, index=False)

        # Store results in database
        session.add(
            BiasMitigationRun(
                upload_id=payload.upload_id,
                sensitive_attribute=",".join(payload.sensitive_columns),
                strategy_used=strategy_name,
                config=payload.strategy_config,
                artifact_model_path=artifact_model_path,
                artifact_dataset_path=artifact_dataset_path,
            )
        )
        await session.commit()

        # ============================================
        # STEP 10: Return comprehensive report
        # ============================================
        return {
            "status": "success",
            "trace_id": trace_id,
            "strategy_applied": strategy_name,
            "metrics_before": baseline_eval,
            "metrics_after": after_eval,
            "fairness_improvement": fairness_improvement,
            "performance_impact": performance_impact,
            "tradeoff_analysis": tradeoff_analysis,
            "artifacts": {
                "corrected_dataset": artifact_dataset_path,
                "model_path": artifact_model_path,
            },
        }

    except MitigationPipelineError as e:
        logger.error(
            "[mitigation:%s] stage=%s error=%s",
            e.trace_id,
            e.stage,
            e.message,
        )
        return _failure_response(e.stage, e.message, e.trace_id)
    except ValueError as e:
        logger.error("[mitigation:%s] Validation error: %s", trace_id, str(e))
        return _failure_response("validation", str(e), trace_id)
    except Exception as e:
        logger.error(
            "[mitigation:%s] Unexpected error: %s\n%s",
            trace_id,
            str(e),
            traceback.format_exc(),
        )
        return _failure_response(
            "unknown", f"Mitigation execution failed: {str(e)}", trace_id
        )



def _stability_score(
    before_metrics: Dict[str, Any], after_metrics: Dict[str, Any]
) -> float:
    deltas = []
    for attr, before_attr in before_metrics["fairness"]["by_attribute"].items():
        after_attr = after_metrics["fairness"]["by_attribute"].get(attr, {})
        if not isinstance(before_attr, dict) or not isinstance(after_attr, dict):
            continue
        before_score = float(
            np.clip(
                1
                - (
                    (
                        abs(before_attr.get("dpd", 0.0))
                        + abs(before_attr.get("eod", 0.0))
                    )
                    / 2
                ),
                0.0,
                1.0,
            )
        )
        after_score = float(
            np.clip(
                1
                - (
                    (abs(after_attr.get("dpd", 0.0)) + abs(after_attr.get("eod", 0.0)))
                    / 2
                ),
                0.0,
                1.0,
            )
        )
        deltas.append(after_score - before_score)

    if not deltas:
        return 0.0

    spread = float(np.std(deltas)) if len(deltas) > 1 else 0.0
    return float(np.clip(1 / (1 + spread), 0.0, 1.0))


def _combined_ranking_score(
    before_summary: Dict[str, Any],
    after_summary: Dict[str, Any],
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
) -> Dict[str, float]:
    before_fairness = before_summary.get("fairness", {}).get("aggregate", {})
    after_fairness = after_summary.get("fairness", {}).get("aggregate", {})
    before_perf = before_summary.get("performance", {})
    after_perf = after_summary.get("performance", {})

    required_fairness_keys = ("fairness_score",)
    required_perf_keys = ("accuracy",)
    for key in required_fairness_keys:
        if key not in before_fairness or key not in after_fairness:
            raise ValueError(
                f"Missing fairness key '{key}' for ranking score computation"
            )
    for key in required_perf_keys:
        if key not in before_perf or key not in after_perf:
            raise ValueError(
                f"Missing performance key '{key}' for ranking score computation"
            )

    fairness_improvement = float(
        np.clip(
            after_fairness["fairness_score"] - before_fairness["fairness_score"],
            0.0,
            1.0,
        )
    )

    before_accuracy = before_perf["accuracy"]
    after_accuracy = after_perf["accuracy"]
    accuracy_retention = 0.0
    if before_accuracy > 0:
        accuracy_retention = float(np.clip(after_accuracy / before_accuracy, 0.0, 1.0))

    stability = _stability_score(before_metrics, after_metrics)
    combined_score = (
        (0.6 * fairness_improvement) + (0.3 * accuracy_retention) + (0.1 * stability)
    )

    return {
        "combined_score": float(np.clip(combined_score, 0.0, 1.0)),
        "fairness_improvement": fairness_improvement,
        "accuracy_retention": accuracy_retention,
        "stability_score": stability,
        "accuracy_drop": float(max(0.0, before_accuracy - after_accuracy)),
    }


def _tradeoff_analysis(strategy_name: str, metrics: Dict[str, float]) -> str:
    fairness_improvement = metrics["fairness_improvement"]
    accuracy_drop = metrics["accuracy_drop"]
    stability = metrics["stability_score"]

    if fairness_improvement <= 0:
        return "No fairness gain detected; mitigation should be reviewed."

    if fairness_improvement > 0.2 and accuracy_drop <= 0.05:
        return "Strong fairness improvement with minimal accuracy reduction."
    if fairness_improvement > 0.2 and accuracy_drop > 0.05:
        return "Strong fairness improvement but with noticeable accuracy trade-off."
    if stability < 0.5:
        return "Fairness gains are inconsistent across sensitive attributes."
    return "Balanced fairness improvement with acceptable performance retention."


def _build_internal_mitigation_payload(payload, strategy_name: str):
    return SimpleNamespace(
        upload_id=payload.upload_id,
        target_column=payload.target_column,
        sensitive_columns=list(payload.sensitive_columns),
        strategy_name=strategy_name,
        strategy_config={},
        confirm_recommendation=True,
    )


# ==================================================================
# PIPELINE-AWARE STRATEGY APPLICATION FUNCTIONS
# ==================================================================


def _apply_reweighting_for_mitigation(
    X_train, y_train, model, sensitive_columns, sensitive_columns_eval, metrics_before
) -> object:
    """
    Apply reweighting strategy for mitigation.
    Computes sample weights for biased attributes and refits model.

    Args:
        X_train: Training features
        y_train: Training labels
        model: Original model
        sensitive_columns: Original sensitive column names
        sensitive_columns_eval: Evaluated sensitive column names (with binning)
        metrics_before: Baseline metrics for each sensitive attribute

    Returns:
        Reweighted model
    """
    combined_weights = None

    for orig_col, eval_col in zip(sensitive_columns, sensitive_columns_eval):
        if eval_col not in metrics_before["fairness"]["by_attribute"]:
            continue

        metrics = metrics_before["fairness"]["by_attribute"][eval_col]

        # Check if mitigation is needed for this attribute
        dpd = abs(metrics.get("dpd", 0.0))
        eod = abs(metrics.get("eod", 0.0))
        dir_ratio = metrics.get("dir", 1.0)

        if dpd <= 0.10 and eod <= 0.10 and dir_ratio >= 0.80:
            logger.debug(f"Attribute '{eval_col}' has low bias, skipping reweighting")
            continue

        if eval_col not in X_train.columns:
            continue

        sensitive_train = X_train[eval_col]
        weights = compute_sample_weights(sensitive_train)

        if combined_weights is None:
            combined_weights = weights
        else:
            combined_weights = combined_weights * weights

        logger.debug(f"Applied reweighting for attribute '{eval_col}'")

    # Fit model with combined weights
    if combined_weights is not None:
        try:
            if isinstance(model, Pipeline):
                final_step_name = model.steps[-1][0]
                fit_params = {f"{final_step_name}__sample_weight": combined_weights}
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, y_train, sample_weight=combined_weights)
            logger.info("Reweighting strategy applied successfully")
        except TypeError as e:
            raise ValueError(
                "Reweighting failed: model may not support sample_weight parameter"
            ) from e
    else:
        logger.info("No attributes required reweighting mitigation")

    return model


def _apply_threshold_for_mitigation(
    X_train,
    y_train,
    model,
    sensitive_columns,
    sensitive_columns_eval,
    metrics_before,
    strategy_config,
) -> object:
    """
    Apply threshold optimization strategy for mitigation.
    Applies ThresholdOptimizer to the most biased attribute.

    Args:
        X_train: Training features
        y_train: Training labels
        model: Original model
        sensitive_columns: Original sensitive column names
        sensitive_columns_eval: Evaluated sensitive column names
        metrics_before: Baseline metrics
        strategy_config: Strategy configuration with grid_size

    Returns:
        ThresholdOptimizer-wrapped model
    """
    applied_attr = None

    # Find most biased attribute by DPD+EOD
    max_bias = 0.0
    for orig_col, eval_col in zip(sensitive_columns, sensitive_columns_eval):
        if eval_col not in metrics_before["fairness"]["by_attribute"]:
            continue

        metrics = metrics_before["fairness"]["by_attribute"][eval_col]
        bias_magnitude = abs(metrics.get("dpd", 0.0)) + abs(metrics.get("eod", 0.0))

        if bias_magnitude > max_bias:
            max_bias = bias_magnitude
            applied_attr = eval_col

    if applied_attr and applied_attr in X_train.columns:
        sensitive_train = X_train[applied_attr]
        grid_size = strategy_config.get("grid_size", 200)

        try:
            model = apply_threshold_optimizer(
                model=model,
                X_train=X_train,
                y_train=y_train,
                sensitive_train=sensitive_train,
                grid_size=grid_size,
            )
            logger.info(f"Threshold optimization applied to attribute '{applied_attr}'")
        except Exception as e:
            raise ValueError(f"Threshold optimization failed: {str(e)}") from e
    else:
        logger.warning("No suitable attribute found for threshold optimization")

    return model


def _apply_smote_for_mitigation(
    X_train, y_train, model, sensitive_columns, sensitive_columns_eval, metrics_before
) -> object:
    """
    Apply SMOTE strategy for mitigation.
    Resamples data for each biased attribute iteratively.

    Args:
        X_train: Training features
        y_train: Training labels
        model: Original model
        sensitive_columns: Original sensitive column names
        sensitive_columns_eval: Evaluated sensitive column names
        metrics_before: Baseline metrics

    Returns:
        SMOTE-resampled model
    """
    current_model = model

    for orig_col, eval_col in zip(sensitive_columns, sensitive_columns_eval):
        if eval_col not in metrics_before["fairness"]["by_attribute"]:
            continue

        metrics = metrics_before["fairness"]["by_attribute"][eval_col]

        # Check if mitigation is needed
        dpd = abs(metrics.get("dpd", 0.0))
        eod = abs(metrics.get("eod", 0.0))
        dir_ratio = metrics.get("dir", 1.0)

        if dpd <= 0.10 and eod <= 0.10 and dir_ratio >= 0.80:
            logger.debug(f"Attribute '{eval_col}' has low bias, skipping SMOTE")
            continue

        if eval_col not in X_train.columns:
            continue

        sensitive_train = X_train[eval_col]

        try:
            current_model, rows_after = apply_smote(
                X_train, y_train, sensitive_train, current_model
            )
            logger.info(
                f"SMOTE applied for attribute '{eval_col}', rows after: {rows_after}"
            )
        except Exception as e:
            logger.error(f"SMOTE failed for attribute '{eval_col}': {str(e)}")
            raise ValueError(f"SMOTE mitigation failed: {str(e)}") from e

    return current_model


async def run_mitigation_ranking(payload, session):
    trace_id = str(uuid.uuid4())
    timeout_seconds = getattr(settings, "MITIGATION_RANK_TIMEOUT_SECONDS", 900)

    strategy_outputs: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    logger.info(
        "[mitigation-rank:%s] Starting ranking for upload_id=%s target=%s sensitive=%s",
        trace_id,
        payload.upload_id,
        payload.target_column,
        payload.sensitive_columns,
    )

    for strategy_name in MITIGATION_STRATEGIES:
        internal_payload = _build_internal_mitigation_payload(payload, strategy_name)

        try:
            output = await asyncio.wait_for(
                run_bias_mitigation(internal_payload, session), timeout=timeout_seconds
            )
            if output.get("status") != "success":
                failures.append(
                    {
                        "strategy": strategy_name,
                        "error_stage": output.get(
                            "error_stage", "mitigation_execution"
                        ),
                        "diagnostic_details": output.get(
                            "diagnostic_details", "Mitigation strategy failed"
                        ),
                        "trace_id": output.get("trace_id"),
                    }
                )
                continue

            strategy_outputs[strategy_name] = output
        except Exception as exc:
            failures.append(
                {
                    "strategy": strategy_name,
                    "error_stage": "strategy_execution",
                    "diagnostic_details": str(exc),
                    "trace_id": trace_id,
                }
            )

    if len(strategy_outputs) != len(MITIGATION_STRATEGIES):
        message = "One or more mitigation strategies failed. Ranking aborted to avoid misleading metrics."
        logger.error("[mitigation-rank:%s] %s failures=%s", trace_id, message, failures)
        return {
            "status": "failed",
            "error_stage": "ranking_execution",
            "diagnostic_details": message,
            "trace_id": trace_id,
            "completed_strategies": list(strategy_outputs.keys()),
            "failed_strategies": failures,
        }

    ranked = []
    consistency_warnings: List[str] = []
    try:
        for strategy_name, output in strategy_outputs.items():
            before_metrics = output.get("metrics_before")
            after_metrics = output.get("metrics_after")
            if not before_metrics or not after_metrics:
                raise MitigationPipelineError(
                    "metric_computation",
                    f"Missing metrics for strategy '{strategy_name}'",
                    trace_id,
                )

            before_summary = before_metrics
            after_summary = after_metrics
            score_bundle = _combined_ranking_score(
                before_summary,
                after_summary,
                before_metrics,
                after_metrics,
            )

            ranked.append(
                {
                    "strategy": strategy_name,
                    "metrics_before": before_summary,
                    "metrics_after": after_summary,
                    **score_bundle,
                    "tradeoff_analysis": _tradeoff_analysis(
                        strategy_name, score_bundle
                    ),
                }
            )

        # Baseline fairness should be nearly identical for all strategies.
        if ranked:
            reference = ranked[0]["metrics_before"]["fairness"]
            for entry in ranked[1:]:
                check = validate_metric_consistency(
                    reference,
                    entry["metrics_before"]["fairness"],
                    tolerance=0.01,
                )
                if not check["consistent"]:
                    msg = (
                        f"Baseline inconsistency between '{ranked[0]['strategy']}' and "
                        f"'{entry['strategy']}': {check['warnings']}"
                    )
                    logger.warning("[mitigation-rank:%s] %s", trace_id, msg)
                    consistency_warnings.append(msg)
    except Exception as exc:
        logger.error(
            "[mitigation-rank:%s] Metric computation failed: %s\n%s",
            trace_id,
            str(exc),
            traceback.format_exc(),
        )
        return _failure_response("metric_computation", str(exc), trace_id)

    ranked.sort(key=lambda item: item["combined_score"], reverse=True)

    best_strategy = ranked[0]["strategy"] if ranked else None

    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        item["recommended"] = index == 1

    ranking_record_rows = []
    for item in ranked:
        ranking_record_rows.append(
            MitigationRanking(
                upload_id=payload.upload_id,
                strategy=item["strategy"],
                metrics_before=item["metrics_before"],
                metrics_after=item["metrics_after"],
                ranking_score=item["combined_score"],
                rank_position=item["rank"],
                status="success",
            )
        )

    for row in ranking_record_rows:
        session.add(row)
    await session.commit()

    summary = (
        f"{best_strategy.capitalize()} achieved the best fairness-performance balance across evaluated strategies."
        if best_strategy
        else "No best strategy could be determined."
    )

    return {
        "status": "success",
        "trace_id": trace_id,
        "baseline_metrics": ranked[0]["metrics_before"] if ranked else {},
        "ranked_strategies": ranked,
        "best_strategy": best_strategy,
        "summary": summary,
        "consistency_warnings": consistency_warnings,
    }
