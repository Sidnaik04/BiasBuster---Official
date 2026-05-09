from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session
from app.schemas.bias import (
    BiasDetectRequest,
    BiasExplanation,
    CorrectBiasRequest,
    CorrectBiasResponse,
    StrategyRecommendationRequest,
)
from app.schemas.bias_mitigation import MitigationRankingRequest
from app.services.bias_service import (
    run_bias_detection,
    apply_bias_correction,
    run_strategy_recommendation,
)
from app.services.bias_mitigation_service import run_mitigation_ranking
from app.utils.mitigation.strategy_evaluator import (
    generate_comparison_report,
)

router = APIRouter(prefix="/api/bias", tags=["Bias Detection"])


@router.post("/detect")
async def detect_bias(
    payload: BiasDetectRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await run_bias_detection(payload, session)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias detection failed: {e}")


@router.post("/recommend-strategy")
async def recommend_mitigation_strategy(
    payload: StrategyRecommendationRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await run_strategy_recommendation(payload, session)
    except KeyError as ke:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bias detection output format: missing {str(ke)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Strategy recommendation failed: {str(e)}"
        )


@router.post("/rank-strategies")
async def rank_mitigation_strategies(
    payload: MitigationRankingRequest,
    session: AsyncSession = Depends(get_session),
):
    """Compatibility wrapper for the new mitigation ranking endpoint."""
    try:
        return await run_mitigation_ranking(payload, session)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Strategy ranking failed: {str(e)}"
        )


@router.post("/correct", response_model=CorrectBiasResponse)
async def correct_bias(
    payload: CorrectBiasRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Apply selected bias mitigation strategies to a dataset and evaluate their effectiveness.

    This endpoint:
    1. Loads the dataset from the specified upload_id
    2. Applies each requested mitigation strategy
    3. Evaluates fairness improvements for each strategy
    4. Returns comparative results with recommendations

    Args:
        upload_id: ID of the uploaded dataset to correct
        target_column: Target label column for the model
        sensitive_columns: List of sensitive attributes to address
        strategy_ids: IDs of strategies to apply (1=threshold, 2=reweighting, 3=smote)

    Returns:
        `CorrectBiasResponse` containing:
        - Results for each applied strategy with before/after metrics
        - Best performing strategy ID
        - Overall summary and recommendations
    """
    try:
        result = await apply_bias_correction(payload, session)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias correction failed: {e}")
