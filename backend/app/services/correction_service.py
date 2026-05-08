"""Service layer for data correction and debiasing."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.models import UploadRecord
from app.utils.dataset_loader import load_dataset
from app.utils.model_loader import load_model
from app.utils.prediction import predict_labels
from app.utils.target_encoder import encode_target_column
from app.utils.feature_encoder import encode_features_for_inference
from app.utils.sensitive_preprocessing import bin_age_column
from app.utils.correction.data_correction_wizard import DataCorrectionWizard
from app.schemas.correction import DataCorrectionWizardRequest


async def run_data_correction_wizard(
    payload: DataCorrectionWizardRequest,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Orchestrate the data correction wizard.

    Flow:
    1. Load dataset and model
    2. Validate and preprocess data
    3. Generate predictions
    4. Initialize wizard
    5. Apply selected strategy
    6. Return results with artifact paths

    Args:
        payload: Correction wizard request
        session: Database session

    Returns:
        Dictionary with correction results
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
    try:
        df = load_dataset(record.dataset_filename)
        model = load_model(record.model_filename)
    except Exception as e:
        raise ValueError(f"Failed to load dataset or model: {str(e)}")

    # -------------------------------------------------
    # STEP 3: Encode target column
    # -------------------------------------------------
    try:
        df, target_info = encode_target_column(df, payload.target_column)
    except Exception as e:
        raise ValueError(f"Target column encoding failed: {str(e)}")

    y_true = df[payload.target_column].astype(int)

    # -------------------------------------------------
    # STEP 4: Handle sensitive columns
    # -------------------------------------------------
    sensitive_cols = payload.sensitive_columns.copy()

    # Age binning if needed
    for col in sensitive_cols:
        if col.lower() == "age":
            df = bin_age_column(df, col)
            sensitive_cols = [c if c != col else col + "_group" for c in sensitive_cols]
            break

    # Convert to string for consistent handling
    for col in sensitive_cols:
        df[col] = df[col].astype(str)

    # -------------------------------------------------
    # STEP 5: Generate predictions
    # -------------------------------------------------
    try:
        X = df.drop(columns=[payload.target_column])

        # Handle pipeline models
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            X_infer = X
        else:
            X_infer = encode_features_for_inference(X)

        y_pred = predict_labels(model, X_infer)
        y_pred = np.nan_to_num(y_pred).astype(int)
    except Exception as e:
        raise ValueError(f"Prediction generation failed: {str(e)}")

    # -------------------------------------------------
    # STEP 6: Initialize and run wizard
    # -------------------------------------------------
    try:
        wizard = DataCorrectionWizard(
            dataset=df,
            model=model,
            target_column=payload.target_column,
            sensitive_columns=sensitive_cols,
            y_predictions=y_pred,
            y_true=y_true,
        )

        result = wizard.correct(strategy=payload.strategy)

        # Ensure all values are JSON-serializable
        _ensure_serializable(result)

        return result

    except Exception as e:
        raise ValueError(f"Data correction wizard failed: {str(e)}")


def _ensure_serializable(obj):
    """Recursively convert non-serializable numpy types to Python types."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _ensure_serializable(value)
    elif isinstance(obj, (list, tuple)):
        return [_ensure_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj
