import os
import joblib
import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import select
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

from app.utils.mitigation.smote import apply_smote
from app.utils.mitigation.reweighting import compute_sample_weights
from app.utils.mitigation.threshold import apply_threshold_optimizer
from app.utils.mitigation.recommender import recommend_strategy
from app.utils.mitigation.strategy_ranker import find_best_strategy

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


async def run_bias_mitigation(payload, session):
    """
    Executes iterative bias mitigation for MULTIPLE sensitive attributes.

    Flow:
    1. Load dataset and model
    2. Run baseline bias detection
    3. Iteratively apply mitigation to each biased attribute
    4. Skip low-bias attributes
    5. Track before/after metrics
    """

    # Fetch upload record
    record = (
        await session.execute(
            select(UploadRecord).where(UploadRecord.id == payload.upload_id)
        )
    ).scalar_one_or_none()

    if not record:
        raise ValueError("Upload record not found")

    # Get dataset and model paths (checks TEMP_DIR first, then ARTIFACT_DIR)
    dataset_path = _get_dataset_path(record.dataset_filename)
    model_path = _get_model_path(record.model_filename)

    df = load_dataset(dataset_path)
    model, preprocessor = load_model(model_path)

    if payload.target_column not in df.columns:
        raise ValueError("Target column not found in dataset")

    # Get test size from config
    test_size = payload.strategy_config.get("test_size", 0.2)

    # Encode target
    df, _ = encode_target_column(df, payload.target_column)
    y = df[payload.target_column]
    X_raw = df.drop(columns=[payload.target_column])

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42, stratify=y
    )

    # Get baseline predictions for fairness evaluation
    y_train_pred = model.predict(X_train_raw)

    # Evaluate baseline metrics
    current_audit = {}
    step_log = []
    artifact_dataset_path = None
    strategy = payload.chosen_strategy

    # Process each sensitive attribute to gather metrics
    for attr in payload.sensitive_attributes:
        eval_attr = "age_group" if attr == "age_group" else attr

        if attr == "age_group" and "age" in X_train_raw.columns:
            X_train_raw = bin_age_column(X_train_raw, "age")

        if eval_attr not in X_train_raw.columns:
            continue

        sensitive_train = X_train_raw[eval_attr]
        baseline_eval = evaluate_baseline(y_train, y_train_pred, sensitive_train)

        current_audit[eval_attr] = {
            "dpd": baseline_eval["fairness"]["dpd"],
            "eod": baseline_eval["fairness"]["eod"],
            "dir": baseline_eval["fairness"]["dir"],
            "performance": baseline_eval["performance"],
        }

    # Execute mitigation strategy
    if strategy == "reweighting":
        current_model = _apply_reweighting_strategy(
            X_train_raw, y_train, model, payload, current_audit, step_log
        )
        model = current_model

        # Save mitigated dataset
        mitigated_df = X_train_raw.copy()
        mitigated_df[payload.target_column] = y_train
        artifact_dataset_path = os.path.join(
            ART_DATA_DIR, f"mitigated_{record.dataset_filename}"
        )
        mitigated_df.to_csv(artifact_dataset_path, index=False)

    elif strategy == "threshold":
        current_model = _apply_threshold_strategy(
            X_train_raw, y_train, model, payload, current_audit, step_log
        )
        model = current_model

    elif strategy == "smote":
        current_model = _apply_smote_strategy(
            X_train_raw, y_train, model, payload, current_audit, step_log
        )
        model = current_model

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Evaluate final metrics
    # Handle ThresholdOptimizer which requires sensitive_features for prediction
    if isinstance(model, ThresholdOptimizer):
        # ThresholdOptimizer needs sensitive_features - use the most biased attribute
        for attr in payload.sensitive_attributes:
            eval_attr = "age_group" if attr == "age_group" else attr
            if eval_attr in X_test_raw.columns:
                sensitive_test = X_test_raw[eval_attr]
                y_test_pred = model.predict(
                    X_test_raw, sensitive_features=sensitive_test
                )
                break
    else:
        y_test_pred = model.predict(X_test_raw)

    after_audit = {}
    for attr in payload.sensitive_attributes:
        eval_attr = "age_group" if attr == "age_group" else attr

        if eval_attr not in X_test_raw.columns:
            continue

        sensitive_test = X_test_raw[eval_attr]
        final_eval = evaluate_baseline(y_test, y_test_pred, sensitive_test)

        after_audit[eval_attr] = {
            "dpd": final_eval["fairness"]["dpd"],
            "eod": final_eval["fairness"]["eod"],
            "dir": final_eval["fairness"]["dir"],
            "performance": final_eval["performance"],
        }

    # Save model
    model_path = os.path.join(ART_MODEL_DIR, f"mitigated_{record.model_filename}")
    joblib.dump(model, model_path)

    # Store results
    session.add(
        BiasMitigationRun(
            upload_id=payload.upload_id,
            sensitive_attribute=",".join(payload.sensitive_attributes),
            strategy_used=strategy,
            config=payload.strategy_config,
            artifact_model_path=model_path,
            artifact_dataset_path=artifact_dataset_path,
        )
    )

    await session.commit()

    return {
        "status": "success",
        "sensitive_attributes": payload.sensitive_attributes,
        "strategy_used": strategy,
        "step_log": step_log,
        "before_metrics": current_audit,
        "after_metrics": after_audit,
        "artifact_model": model_path,
        "artifact_dataset": artifact_dataset_path,
    }


def _apply_reweighting_strategy(
    X_train, y_train, model, payload, baseline_audit, step_log
):
    """
    Apply iterative reweighting: multiply weights cumulatively then fit once.
    """
    combined_weights = None
    applied_attrs = []

    for attr in payload.bias_ranking:
        eval_attr = "age_group" if attr == "age_group" else attr

        if eval_attr not in baseline_audit:
            continue

        metrics = baseline_audit[eval_attr]
        if not _requires_mitigation(metrics):
            step_log.append({"attribute": attr, "applied": False, "reason": "low bias"})
            continue

        applied_attrs.append(attr)

        if eval_attr not in X_train.columns:
            continue

        sensitive_train = X_train[eval_attr]
        weights = compute_sample_weights(sensitive_train)

        if combined_weights is None:
            combined_weights = weights
        else:
            combined_weights = combined_weights * weights

        step_log.append(
            {"attribute": attr, "applied": True, "fairness_metrics": metrics}
        )

    # Apply combined weights in a single fit
    if combined_weights is not None:
        try:
            if isinstance(model, Pipeline):
                final_step_name = model.steps[-1][0]
                fit_params = {f"{final_step_name}__sample_weight": combined_weights}
                model.fit(X_train, y_train, **fit_params)
            else:
                model.fit(X_train, y_train, sample_weight=combined_weights)
        except Exception as e:
            raise ValueError(
                "Reweighting failed: model may not support sample_weight"
            ) from e
    else:
        step_log.append({"message": "No biased attributes found, model not modified"})

    return model


def _apply_threshold_strategy(
    X_train, y_train, model, payload, baseline_audit, step_log
):
    """
    Apply threshold optimization to the most biased attribute only.
    """
    applied_attr = None

    # Find most biased attribute
    for attr in payload.bias_ranking:
        eval_attr = "age_group" if attr == "age_group" else attr

        if eval_attr not in baseline_audit:
            continue

        metrics = baseline_audit[eval_attr]
        if _requires_mitigation(metrics):
            applied_attr = attr
            break

    if applied_attr:
        eval_attr = "age_group" if applied_attr == "age_group" else applied_attr

        if eval_attr in X_train.columns:
            sensitive_train = X_train[eval_attr]
            model = apply_threshold_optimizer(
                model=model,
                X_train=X_train,
                y_train=y_train,
                sensitive_train=sensitive_train,
                grid_size=payload.strategy_config.get("grid_size", 200),
            )
            step_log.append(
                {
                    "attribute": applied_attr,
                    "applied": True,
                    "reason": "Most biased attribute selected for threshold optimizer",
                }
            )

    # Log skipped attributes
    for attr in payload.bias_ranking:
        if attr != applied_attr:
            step_log.append(
                {
                    "attribute": attr,
                    "applied": False,
                    "reason": "ThresholdOptimizer applied only once to most biased attribute",
                }
            )

    return model


def _apply_smote_strategy(X_train, y_train, model, payload, baseline_audit, step_log):
    """
    Apply iterative SMOTE: resample for each biased attribute sequentially.
    """
    current_model = model

    for attr in payload.bias_ranking:
        eval_attr = "age_group" if attr == "age_group" else attr

        if eval_attr not in baseline_audit:
            continue

        metrics = baseline_audit[eval_attr]
        if not _requires_mitigation(metrics):
            step_log.append({"attribute": attr, "applied": False, "reason": "low bias"})
            continue

        if eval_attr not in X_train.columns:
            continue

        sensitive_train = X_train[eval_attr]
        current_model, rows_after = apply_smote(
            X_train, y_train, sensitive_train, current_model
        )

        step_log.append(
            {
                "attribute": attr,
                "applied": True,
                "rows_after_smote": rows_after,
                "fairness_metrics": metrics,
            }
        )

    return current_model
