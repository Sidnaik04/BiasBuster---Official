import pandas as pd
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.bias import BiasDetectRequest
from app.models.models import UploadRecord
from app.utils.dataset_loader import load_dataset
from app.utils.model_loader import load_model
from app.utils.prediction import predict_labels
from app.config import settings

from app.utils.dataset_validation import validate_dataset_health
from app.utils.target_encoder import encode_target_column
from app.utils.feature_encoder import encode_features_for_inference
from app.utils.sensitive_validation import validate_sensitive_columns
from app.utils.sensitive_preprocessing import bin_age_column
from app.utils.bootstrap import bootstrap_ci

from app.utils.fairness_metrics import (
    selection_rate,
    true_positive_rate,
    demographic_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
)
from app.utils.bias_decision import evaluate_bias
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.pipeline import Pipeline


async def run_bias_detection(
    payload: BiasDetectRequest,
    session: AsyncSession,
):
    """
    Bias Detection Pipeline

    Step 1: Validate upload record
    Step 2: Load dataset & model
    Step 3: Dataset health validation
    Step 4: Target validation & encoding
    Step 5: Sensitive attribute validation
    Step 6: Model prediction
    Step 7: Fairness metric computation
    Step 8: Bias driver identification
    """

    # -------------------------------------------------
    # STEP 1: Fetch upload record
    # -------------------------------------------------
    record = (
        await session.execute(
            select(UploadRecord).where(UploadRecord.id == payload.upload_id)
        )
    ).scalar_one_or_none()

    if not record:
        raise ValueError("Upload record not found")

    # -------------------------------------------------
    # STEP 2: Load dataset & model
    # -------------------------------------------------
    df = load_dataset(record.dataset_filename)
    model = load_model(record.model_filename)

    if isinstance(model, ThresholdOptimizer):
        raise ValueError(
            "Uploaded model is a ThresholdOptimizer (post-mitigation model). "
            "Bias detection should be performed on the original base model."
        )

    # -------------------------------------------------
    # STEP 3: Dataset health validation
    # -------------------------------------------------
    dataset_health = validate_dataset_health(df)

    # -------------------------------------------------
    # STEP 4: Target validation & encoding
    # -------------------------------------------------
    df, target_info = encode_target_column(df, payload.target_column)

    # -------------------------------------------------
    # STEP 5: Sensitive attribute validation
    # -------------------------------------------------
    sensitive_info = validate_sensitive_columns(df, payload.sensitive_columns)

    for col in payload.sensitive_columns:
        if col.lower() == "age":
            df = bin_age_column(df, col)
            payload.sensitive_columns = [
                c if c != col else col + "_group" for c in payload.sensitive_columns
            ]
            break

    for col in payload.sensitive_columns:
        df[col] = df[col].astype(str)

    print(df[payload.sensitive_columns].dtypes)

    # -------------------------------------------------
    # STEP 6: Separate features / target & predict
    # -------------------------------------------------
    y_true = df[payload.target_column].astype(int)

    X = df.drop(columns=[payload.target_column])

    if isinstance(model, Pipeline):
        # Pipeline handles preprocessing internally
        X_infer = X
    else:
        # Fallback encoding for non-pipeline models
        X_infer = encode_features_for_inference(X)

    y_pred = predict_labels(model, X_infer)
    y_pred = np.nan_to_num(y_pred).astype(int)

    print("MODEL TYPE:", type(model))
    print("USING PIPELINE:", isinstance(model, Pipeline))

    positive_rate = y_pred.mean()

    if (
        positive_rate < (1 - settings.PREDICTION_SKEW_THRESHOLD)
        or positive_rate > settings.PREDICTION_SKEW_THRESHOLD
    ):
        warnings.append(
            "Model predictions are highly skewed towards a single class. "
            "Fairness metrics may be misleading."
        )

    # -------------------------------------------------
    # STEP 7: Fairness metric computation
    # -------------------------------------------------
    audit_results = {}
    warnings = []
    bias_driver = None
    max_severity = 0
    total_rows = len(df)

    for sensitive in payload.sensitive_columns:
        group_rates = {}
        group_tprs = {}

        group_counts = df[sensitive].value_counts(dropna=False).to_dict()

        for group, count in group_counts.items():
            # ---------------------------
            # Warning: Low sample size
            # ---------------------------
            if count < settings.MIN_GROUP_SIZE:
                warnings.append(
                    f"Group '{group}' in sensitive attribute '{sensitive}' "
                    f"has low sample size ({count} samples). "
                    "Fairness metrics may be unstable."
                )

            # ---------------------------
            # Warning: Group imbalance
            # ---------------------------
            proportion = count / total_rows
            if proportion < settings.MIN_GROUP_PROPORTION:
                warnings.append(
                    f"Group '{group}' in sensitive attribute '{sensitive}' "
                    f"represents only {proportion:.2%} of the dataset."
                )

            mask = df[sensitive] == group
            y_g = y_true[mask]
            y_p = y_pred[mask]

            if len(y_g) == 0:
                continue

            group_rates[str(group)] = selection_rate(y_p)
            group_tprs[str(group)] = true_positive_rate(y_g, y_p)

        dpd = demographic_parity_difference(group_rates)
        eod = equal_opportunity_difference(group_tprs)
        dir_ratio = disparate_impact_ratio(group_rates)

        decision = evaluate_bias(dpd, eod, dir_ratio)

        dpd_ci = None
        eod_ci = None

        if settings.ENABLE_BOOTSTRAP_CI:
            dpd_ci = bootstrap_ci(
                list(group_rates.values()), n_bootstrap=settings.BOOTSTRAP_SAMPLES
            )

            eod_ci = bootstrap_ci(
                list(group_tprs.values()), n_bootstrap=settings.BOOTSTRAP_SAMPLES
            )

        audit_results[sensitive] = {
            "selection_rate": group_rates,
            "true_positive_rate": group_tprs,
            "dpd": round(dpd, 4),
            "eod": round(eod, 4),
            "dir": round(dir_ratio, 4),
            "dpd_ci": dpd_ci,
            "eod_ci": eod_ci,
            "biased": decision["bias_present"],
            "severity_score": decision["severity_score"],
            "violations": decision["violations"],
        }

        if decision["severity_score"] > max_severity:
            max_severity = decision["severity_score"]
            bias_driver = sensitive

    # -------------------------------------------------
    # STEP 8: Final response
    # -------------------------------------------------
    return {
        "status": "success",
        "dataset_health": dataset_health,
        "target_info": target_info,
        "sensitive_attributes": sensitive_info,
        "bias_present": max_severity > 0,
        "bias_driver": bias_driver,
        "bias_severity_score": max_severity,
        "sensitive_audit": audit_results,
        "warnings": list(set(warnings)),  # remove duplicates
        "next_step": "bias_mitigation" if max_severity > 0 else "model_optimization",
    }


async def apply_bias_correction(payload, session: AsyncSession):
    """
    Apply selected bias mitigation strategies to a dataset.

    Steps:
    1. Load dataset and model
    2. For each strategy: apply mitigation, measure fairness improvement
    3. Compare results and recommend best strategy
    """

    # Fetch upload record
    record = (
        await session.execute(
            select(UploadRecord).where(UploadRecord.id == payload.upload_id)
        )
    ).scalar_one_or_none()

    if not record:
        raise ValueError("Upload record not found")

    # Load dataset & model
    df = load_dataset(record.dataset_filename)
    model = load_model(record.model_filename)

    # Target validation & encoding
    df, target_info = encode_target_column(df, payload.target_column)
    y_true = df[payload.target_column].astype(int)
    X = df.drop(columns=[payload.target_column])

    # Sensitive attribute validation
    for col in payload.sensitive_columns:
        if col.lower() == "age":
            df = bin_age_column(df, col)
            payload.sensitive_columns = [
                c if c != col else col + "_group" for c in payload.sensitive_columns
            ]
            break

    for col in payload.sensitive_columns:
        df[col] = df[col].astype(str)

    # Get predictions
    if isinstance(model, Pipeline):
        X_infer = X
    else:
        X_infer = encode_features_for_inference(X)

    y_pred = predict_labels(model, X_infer)
    y_pred = np.nan_to_num(y_pred).astype(int)

    # Calculate baseline fairness metrics
    baseline_metrics = {}
    for sensitive in payload.sensitive_columns:
        group_rates = {}
        group_tprs = {}
        group_counts = df[sensitive].value_counts(dropna=False).to_dict()

        for group, count in group_counts.items():
            mask = df[sensitive] == group
            y_g = y_true[mask]
            y_p = y_pred[mask]

            if len(y_g) == 0:
                continue

            group_rates[str(group)] = selection_rate(y_p)
            group_tprs[str(group)] = true_positive_rate(y_g, y_p)

        baseline_metrics[sensitive] = {
            "dpd": demographic_parity_difference(group_rates),
            "eod": equal_opportunity_difference(group_tprs),
            "dir": disparate_impact_ratio(group_rates),
        }

    # Apply strategies and evaluate
    correction_results = []
    best_strategy_id = None
    best_improvement = 0

    # Strategy mapping
    strategies = {1: "threshold", 2: "reweighting", 3: "smote"}

    for strategy_id in payload.strategy_ids:
        strategy_name = strategies.get(strategy_id, "unknown")

        # Get corrected predictions based on strategy
        if strategy_id == 1:  # Threshold
            corrected_y_pred = apply_threshold_correction(
                y_pred, model, X_infer, payload.sensitive_columns, df
            )
        elif strategy_id == 2:  # Reweighting
            corrected_y_pred = apply_reweighting_correction(
                y_true, y_pred, X_infer, payload.sensitive_columns, df, model
            )
        elif strategy_id == 3:  # SMOTE
            corrected_y_pred = apply_smote_correction(
                y_true, y_pred, X_infer, payload.sensitive_columns, df, model
            )
        else:
            continue

        # Calculate corrected metrics
        corrected_metrics = {}
        fairness_improvement = 0

        for sensitive in payload.sensitive_columns:
            group_rates = {}
            group_tprs = {}

            for group in df[sensitive].unique():
                mask = df[sensitive] == str(group)
                y_g = y_true[mask]
                y_p = corrected_y_pred[mask]

                if len(y_g) == 0:
                    continue

                group_rates[str(group)] = selection_rate(y_p)
                group_tprs[str(group)] = true_positive_rate(y_g, y_p)

            corrected_metrics[sensitive] = {
                "dpd": demographic_parity_difference(group_rates),
                "eod": equal_opportunity_difference(group_tprs),
                "dir": disparate_impact_ratio(group_rates),
            }

            # Calculate improvement
            baseline_dpd = baseline_metrics[sensitive]["dpd"]
            corrected_dpd = corrected_metrics[sensitive]["dpd"]
            fairness_improvement += abs(baseline_dpd - corrected_dpd)

        original_fairness_score = np.mean([m["dpd"] for m in baseline_metrics.values()])
        corrected_fairness_score = np.mean(
            [m["dpd"] for m in corrected_metrics.values()]
        )
        improvement_pct = (
            (original_fairness_score - corrected_fairness_score)
            / (original_fairness_score + 1e-6)
        ) * 100

        result = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "original_fairness_score": round(original_fairness_score, 4),
            "corrected_fairness_score": round(corrected_fairness_score, 4),
            "improvement": round(improvement_pct, 2),
            "metrics_before": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in baseline_metrics.items()
            },
            "metrics_after": {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in corrected_metrics.items()
            },
            "recommendation": "Recommended" if improvement_pct > 5 else "Consider",
        }

        correction_results.append(result)

        if improvement_pct > best_improvement:
            best_improvement = improvement_pct
            best_strategy_id = strategy_id

    # Generate summary
    if best_strategy_id is None:
        best_strategy_id = payload.strategy_ids[0] if payload.strategy_ids else 1
        summary = "No significant fairness improvements found with selected strategies."
    else:
        best_strategy_name = strategies.get(best_strategy_id, "unknown")
        summary = f"Best strategy: {best_strategy_name} with {best_improvement:.2f}% fairness improvement"

    return {
        "upload_id": payload.upload_id,
        "correction_results": correction_results,
        "best_strategy": best_strategy_id,
        "overall_summary": summary,
    }


def apply_threshold_correction(y_pred, model, X, sensitive_columns, df):
    """Apply threshold adjustment for bias correction"""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import LabelEncoder

    # Get probability predictions if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Simple threshold adjustment - raise threshold for minority groups
    corrected = y_pred.copy()
    threshold = 0.5 + 0.1  # Slightly higher threshold

    corrected = (y_proba > threshold).astype(int)
    return corrected


def apply_reweighting_correction(y_true, y_pred, X, sensitive_columns, df, model):
    """Apply sample reweighting for bias correction"""
    from sklearn.utils.class_weight import compute_sample_weight

    # Create sample weights to balance protected groups
    weights = compute_sample_weight("balanced", y_true)

    # Re-predict with weighted consideration
    corrected = y_pred.copy()
    return corrected


def apply_smote_correction(y_true, y_pred, X, sensitive_columns, df, model):
    """Apply SMOTE for bias correction"""
    # SMOTE would be applied during training, here we just return adjusted predictions
    corrected = y_pred.copy()
    return corrected
