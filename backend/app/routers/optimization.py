"""Optimization API endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.optimization import OptimizationRequest, OptimizationResponse
from app.services.optimization_service import run_model_optimization

router = APIRouter(prefix="/api", tags=["Optimization"])


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_model(
    payload: OptimizationRequest,
    session: AsyncSession = Depends(get_session),
) -> OptimizationResponse:
    """
    Optimize model hyperparameters with fairness awareness.

    Supports two optimization methods:
    - **gridsearch**: Exhaustive parameter search (GridSearchCV)
    - **optuna**: Probabilistic optimization (Optuna)

    Combined Score Formula:
    combined_score = (accuracy_weight × accuracy) + (fairness_weight × fairness_score)

    Where:
    fairness_score = 1 - (|DPD| + |EOD|) / 2

    Args:
        payload: Optimization request with:
            - upload_id: ID of uploaded dataset and model
            - target_column: Target variable name
            - sensitive_columns: List of sensitive attributes
            - method: "gridsearch" or "optuna"
            - n_trials: Number of trials (Optuna only)
            - cv_folds: K-fold cross-validation (GridSearch only)
            - accuracy_weight: Weight for accuracy (0-1)
            - fairness_weight: Weight for fairness (0-1)
            - timeout: Max runtime in seconds (Optuna only)

    Returns:
        {
            "status": "success" | "error",
            "best_params": {...},
            "best_score": 0.87,
            "accuracy": 0.91,
            "fairness_score": 0.78,
            "combined_score": 0.87,
            "dpd": 0.15,
            "eod": 0.10,
            "optimization_method": "optuna" | "gridsearch",
            "trials_run": 20,
            "comparison": [
                {
                    "params": {...},
                    "accuracy": 0.89,
                    "fairness_score": 0.75,
                    "combined_score": 0.83,
                    "dpd": 0.18,
                    "eod": 0.12
                },
                ...
            ],
            "accuracy_weight": 0.6,
            "fairness_weight": 0.4,
            "message": "Optimization completed..."
        }

    Examples:
        ```json
        POST /api/optimize
        {
            "upload_id": 1,
            "target_column": "income",
            "sensitive_columns": ["gender", "race"],
            "method": "optuna",
            "n_trials": 30,
            "accuracy_weight": 0.6,
            "fairness_weight": 0.4
        }
        ```
    """

    try:
        result = await run_model_optimization(payload, session)
        # Return the OptimizationResponse directly (FastAPI will convert to dict)
        return result

    except Exception as e:
        import traceback

        print(f"Endpoint error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}",
        )
