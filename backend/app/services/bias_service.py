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
