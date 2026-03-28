from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session
from app.schemas.bias_mitigation import BiasMitigationRequest
from app.services.bias_mitigation_service import run_bias_mitigation

router = APIRouter(prefix="/api/bias", tags=["Bias Mitigation"])


@router.post("/mitigate")
async def mitigate_bias(
    payload: BiasMitigationRequest,
    session: AsyncSession = Depends(get_session),
):
    if not payload.confirm_recommendation:
        raise HTTPException(
            status_code=400, detail="Human-in-the-loop confirmation required"
        )

    try:
        return await run_bias_mitigation(payload, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {e}")
