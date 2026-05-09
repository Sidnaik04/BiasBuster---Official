from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session
from app.schemas.bias_mitigation import BiasMitigationRequest, MitigationRankingRequest
from app.services.bias_mitigation_service import (
    run_bias_mitigation,
    run_mitigation_ranking,
    validate_mitigation_strategy,
)

router = APIRouter(prefix="/api/bias", tags=["Bias Mitigation"])


@router.post("/mitigate")
async def mitigate_bias(
    payload: BiasMitigationRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Execute fully pipeline-aware bias mitigation.

    Workflow:
    1. Load dataset and model from upload_id
    2. Validate strategy_name
    3. If confirm_recommendation=False, return confirmation_required status
    4. Compute baseline fairness metrics internally
    5. Apply mitigation strategy
    6. Compute mitigated metrics
    7. Return comprehensive mitigation report with tradeoff analysis

    Required fields:
    - upload_id: ID of uploaded dataset/model
    - target_column: Column to predict (target)
    - sensitive_columns: List of protected attributes
    - strategy_name: One of ['threshold', 'reweighting', 'smote']
    - confirm_recommendation: Boolean confirmation flag

    Optional fields:
    - strategy_config: Dict with strategy-specific parameters
    """

    # Validate strategy
    try:
        validate_mitigation_strategy(payload.strategy_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Handle confirmation logic
    if not payload.confirm_recommendation:
        return {
            "status": "confirmation_required",
            "message": "Human-in-the-loop confirmation is required to execute mitigation.",
            "strategy_name": payload.strategy_name,
            "upload_id": payload.upload_id,
        }

    try:
        return await run_bias_mitigation(payload, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")


@router.post("/mitigation/rank")
async def rank_mitigations(
    payload: MitigationRankingRequest,
    session: AsyncSession = Depends(get_session),
):
    try:
        return await run_mitigation_ranking(payload, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation ranking failed: {e}")
