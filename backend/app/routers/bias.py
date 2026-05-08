from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session
from app.schemas.bias import (
    BiasDetectRequest,
    BiasExplanation,
    CorrectBiasRequest,
    CorrectBiasResponse,
)
from app.services.bias_service import run_bias_detection, apply_bias_correction
from app.utils.mitigation.strategy_recommender import recommend_strategy
from app.utils.mitigation.strategy_evaluator import (
    find_best_strategy,
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
async def recommend_mitigation_strategy(bias_detection_result: dict):
    """
    Recommend optimal bias mitigation strategy based on bias detection results.

    Args:
        bias_detection_result: Complete output from /api/bias/detect endpoint

    Returns:
        {
            "recommended_strategy": "threshold" | "reweighting" | "smote" | "none",
            "confidence_score": 0.0-1.0,
            "reasoning": str,
            "target_attributes": [str],
            "expected_improvements": {
                "dpd": float (% change expected),
                "eod": float (% change expected),
                "accuracy_impact": float (% change expected)
            },
            "warnings": [str],
            "alternative_strategies": [
                {"strategy": str, "reason": str},
                ...
            ]
        }
    """
    try:
        recommendation = recommend_strategy(bias_detection_result)
        return {"status": "success", "recommendation": recommendation}
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
async def rank_mitigation_strategies(strategy_results: dict):
    """
    Rank mitigation strategies by comparing fairness improvements vs. accuracy trade-offs.

    Call this endpoint AFTER testing all three strategies (threshold, reweighting, smote)
    to get an objective ranking of which worked best.

    Args:
        strategy_results: Results from all tested strategies in this format:
            {
                "reweighting": {
                    "before_metrics": {
                        "fairness": {"dpd": float, "eod": float, "dir": float},
                        "performance": {"accuracy": float, "precision": float, "recall": float, "f1": float}
                    },
                    "after_metrics": {
                        "fairness": {"dpd": float, "eod": float, "dir": float},
                        "performance": {"accuracy": float, "precision": float, "recall": float, "f1": float}
                    }
                },
                "threshold": {...},
                "smote": {...}
            }

        Optional query params:
        - fairness_weight: (default 1.0) How much to value fairness improvements
        - accuracy_weight: (default 0.5) How much to penalize accuracy loss

    Returns:
        {
            "status": "success",
            "best_strategy": "threshold",
            "best_score": 0.8542,
            "ranking": [
                {
                    "rank": 1,
                    "strategy": "threshold",
                    "total_score": 0.8542,
                    "fairness_improvement": 0.1068,
                    "accuracy_impact": -0.0201,
                    "dpd_improvement": 0.1014,
                    "eod_improvement": 0.0926,
                    "di_improvement": 0.3379
                },
                ...
            ],
            "insights": [
                "🏆 THRESHOLD is a clear winner...",
                "📊 Threshold sacrifices..."
            ],
            "recommendation": "✅ Deploy Threshold Optimizer..."
        }
    """
    try:
        # Generate detailed comparison report
        report = generate_comparison_report(strategy_results)
        return report
    except KeyError as ke:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy results format: missing {str(ke)}. "
            f"Expected: {{'strategy_name': {{'before_metrics': {...}, 'after_metrics': {...}}}}}",
        )
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
