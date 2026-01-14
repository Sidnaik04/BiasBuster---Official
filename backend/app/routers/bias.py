from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session
from app.schemas.bias import BiasDetectRequest
from app.services.bias_service import run_bias_detection

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
