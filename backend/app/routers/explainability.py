"""
API Router for Bias Explainability Engine.

Endpoints:
- POST /api/explain/bias - Explain WHY bias occurs in the model
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd

from app.db import get_session
from app.models.models import UploadRecord
from app.schemas.bias import ExplainBiasRequest, BiasExplanation
from app.utils.dataset_loader import load_dataset
from app.utils.model_loader import load_model
from app.utils.explainability import BiasExplainer
from app.utils.feature_encoder import encode_features_for_inference
from app.utils.target_encoder import encode_target_column

router = APIRouter(prefix="/api/explain", tags=["Bias Explainability"])


@router.post("/bias", response_model=BiasExplanation)
async def explain_bias(
    payload: ExplainBiasRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Explain WHY bias occurs in a machine learning model.

    Analyzes:
    - Feature contributions to bias
    - Group-wise disparities
    - Proxy/discriminatory features
    - Overall bias patterns

    Args:
        payload: ExplainBiasRequest with upload_id, target_column, sensitive_columns

    Returns:
        BiasExplanation with detailed findings and human-readable interpretations

    Raises:
        ValueError: If data validation fails
        HTTPException: For server errors
    """
    try:
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

        # -------------------------------------------------
        # STEP 3: Prepare data
        # -------------------------------------------------
        # Encode target column
        df_encoded, target_info = encode_target_column(df, payload.target_column)
        y = df_encoded[payload.target_column]

        # Extract features (exclude target)
        # Keep original X WITHOUT encoding - the model's pipeline will handle it
        X = df_encoded.drop(columns=[payload.target_column])

        # Extract sensitive attributes (before encoding features)
        sensitive_attrs = df[payload.sensitive_columns].copy()

        # -------------------------------------------------
        # STEP 4: Initialize and run BiasExplainer
        # -------------------------------------------------
        # Pass original X, not encoded - the model's ColumnTransformer handles transformation
        explainer = BiasExplainer(
            model=model,
            X=X,
            y=y,
            sensitive_attributes=sensitive_attrs,
            target_column=payload.target_column,
        )

        # Generate complete explanation
        explanation = explainer.explain_bias()

        return BiasExplanation(
            bias_detected=explanation["bias_detected"],
            top_bias_contributors=explanation["top_bias_contributors"],
            group_analysis=explanation["group_analysis"],
            proxy_features=explanation["proxy_features"],
            summary=explanation["summary"],
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Bias explanation failed: {str(e)}"
        )


@router.post("/debug")
async def debug_explain_bias(
    payload: ExplainBiasRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Debug endpoint for bias explanation (returns raw dict).

    Same as /bias but returns raw Python dict for debugging.

    Args:
        payload: ExplainBiasRequest

    Returns:
        Dict with complete explanation data
    """
    try:
        record = (
            await session.execute(
                select(UploadRecord).where(UploadRecord.id == payload.upload_id)
            )
        ).scalar_one_or_none()

        if not record:
            raise ValueError("Upload record not found")

        df = load_dataset(record.dataset_filename)
        model = load_model(record.model_filename)

        df_encoded, target_info = encode_target_column(df, payload.target_column)
        y = df_encoded[payload.target_column]
        X = df_encoded.drop(columns=[payload.target_column])
        sensitive_attrs = df[payload.sensitive_columns].copy()

        explainer = BiasExplainer(
            model=model,
            X=X,
            y=y,
            sensitive_attributes=sensitive_attrs,
            target_column=payload.target_column,
        )

        explanation = explainer.explain_bias()

        # Also compute SHAP if available
        shap_result = explainer.compute_shap_values()
        if shap_result:
            explanation["shap_analysis"] = shap_result

        return {
            "status": "success",
            "data": explanation,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Debug explanation failed: {str(e)}",
        )
