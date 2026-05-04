"""
Experiments API Router: Endpoints for auto experimentation engine.

Endpoints:
- POST /api/experiments/run : Run full experiment pipeline
- GET /api/experiments/history : Retrieve experiment history
- GET /api/experiments/:experiment_id : Get specific experiment details
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import logging

from app.db import get_session
from app.models.models import UploadRecord
from app.models.experiment import ExperimentRun
from app.schemas.experiment import (
    ExperimentRunRequest,
    ExperimentReportResponse,
    ExperimentHistoryResponse,
    ExperimentHistoryListResponse,
)
from app.utils.experiments import ExperimentRunner

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["Experiments"])


@router.post("/run", response_model=ExperimentReportResponse)
async def run_experiment(
    payload: ExperimentRunRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Run full auto experimentation pipeline.

    Automatically tests multiple mitigation strategies (threshold, reweighting, SMOTE)
    and compares their impact on accuracy and fairness.

    Input:
        - upload_id: ID of uploaded dataset/model pair
        - target_column: Name of target/label column
        - sensitive_columns: List of sensitive attribute names
        - strategies: Strategies to test (default: all)
        - timeout_per_strategy: Timeout per strategy in seconds

    Output:
        - experiment_id: Unique experiment identifier
        - strategies_tested: List of tested strategies
        - results: Results for each strategy
        - best_strategy: Strategy with best fairness-accuracy trade-off
        - insights: Generated explanations
        - metrics_before: Baseline metrics

    Example response:
    {
        "experiment_id": "abc123",
        "strategies_tested": ["threshold", "reweighting", "smote"],
        "results": [
            {
                "strategy": "threshold",
                "accuracy_before": 0.91,
                "accuracy_after": 0.88,
                "fairness_score_before": 0.52,
                "fairness_score_after": 0.78,
                "fairness_improvement": 0.26,
                "accuracy_drop": -0.03,
                "combined_score": 0.162
            }
        ],
        "best_strategy": "threshold",
        "insights": "Threshold adjustment provided best fairness gain with minimal accuracy loss."
    }
    """

    try:
        # Validate upload record exists
        record = (
            await session.execute(
                select(UploadRecord).where(UploadRecord.id == payload.upload_id)
            )
        ).scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Upload record not found")

        logger.info(f"Starting experiment for upload_id={payload.upload_id}")

        # Initialize experiment runner
        runner = ExperimentRunner(
            dataset_filename=record.dataset_filename,
            model_filename=record.model_filename,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
        )

        # Run experiment pipeline
        experiment_report = runner.run_experiment(
            strategies=payload.strategies,
            timeout_per_strategy=payload.timeout_per_strategy,
        )

        # Store experiment results in database
        db_experiment = ExperimentRun(
            experiment_id=experiment_report["experiment_id"],
            upload_id=payload.upload_id,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
            strategies_tested=experiment_report["strategies_tested"],
            metrics_before=experiment_report["metrics_before"],
            results=experiment_report["results"],
            best_strategy=experiment_report["best_strategy"],
            combined_score=experiment_report.get("best_strategy_score"),
            insights=experiment_report["insights"],
            total_duration_seconds=experiment_report["total_duration_seconds"],
            status=experiment_report["status"],
            error_message=experiment_report.get("error_message"),
        )

        session.add(db_experiment)
        await session.commit()
        await session.refresh(db_experiment)

        logger.info(f"Experiment stored: {experiment_report['experiment_id']}")

        # Build response
        response = ExperimentReportResponse(
            experiment_id=experiment_report["experiment_id"],
            upload_id=payload.upload_id,
            target_column=payload.target_column,
            sensitive_columns=payload.sensitive_columns,
            metrics_before=experiment_report["metrics_before"],
            strategies_tested=experiment_report["strategies_tested"],
            results=[
                {
                    **result,
                    "status": result.get("status", "success"),
                    "error": result.get("error"),
                }
                for result in experiment_report["results"]
            ],
            best_strategy=experiment_report["best_strategy"],
            best_strategy_score=experiment_report.get("best_strategy_score"),
            insights=experiment_report["insights"],
            warnings=experiment_report.get("warnings", []),
            total_duration_seconds=experiment_report["total_duration_seconds"],
            status=experiment_report["status"],
            created_at=db_experiment.created_at,
        )

        return response

    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}")


@router.get("/history", response_model=ExperimentHistoryListResponse)
async def get_experiment_history(
    upload_id: int = Query(None, description="Filter by upload_id"),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset"),
    session: AsyncSession = Depends(get_session),
):
    """
    Retrieve experiment history.

    Query parameters:
        - upload_id: Optional filter by specific upload
        - limit: Max results (default 10, max 100)
        - offset: Pagination offset (default 0)

    Returns list of experiments with summary info.
    """

    try:
        # Build query
        query = select(ExperimentRun)

        if upload_id:
            query = query.where(ExperimentRun.upload_id == upload_id)

        # Order by created_at descending
        query = query.order_by(desc(ExperimentRun.created_at))

        # Count total
        count_query = select(ExperimentRun)
        if upload_id:
            count_query = count_query.where(ExperimentRun.upload_id == upload_id)

        count_result = await session.execute(
            select(func.count(ExperimentRun.id)).select_from(ExperimentRun)
        )
        if upload_id:
            count_result = await session.execute(
                select(func.count(ExperimentRun.id))
                .select_from(ExperimentRun)
                .where(ExperimentRun.upload_id == upload_id)
            )

        total = count_result.scalar() or 0

        # Apply pagination
        query = query.limit(limit).offset(offset)

        # Execute query
        result = await session.execute(query)
        experiments = result.scalars().all()

        # Convert to response models
        history = [
            ExperimentHistoryResponse(
                experiment_id=exp.experiment_id,
                upload_id=exp.upload_id,
                target_column=exp.target_column,
                best_strategy=exp.best_strategy,
                best_strategy_score=exp.combined_score,
                created_at=exp.created_at,
                status=exp.status,
            )
            for exp in experiments
        ]

        return ExperimentHistoryListResponse(
            total=total,
            experiments=history,
        )

    except Exception as e:
        logger.error(f"Failed to retrieve experiment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentReportResponse)
async def get_experiment_detail(
    experiment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get detailed report for specific experiment.

    Path parameters:
        - experiment_id: Unique experiment identifier

    Returns full experiment report with all strategy results and metrics.
    """

    try:
        # Fetch experiment
        result = await session.execute(
            select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
        )
        experiment = result.scalar_one_or_none()

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Build response
        response = ExperimentReportResponse(
            experiment_id=experiment.experiment_id,
            upload_id=experiment.upload_id,
            target_column=experiment.target_column,
            sensitive_columns=experiment.sensitive_columns,
            metrics_before=experiment.metrics_before,
            strategies_tested=experiment.strategies_tested,
            results=experiment.results,
            best_strategy=experiment.best_strategy,
            best_strategy_score=experiment.combined_score,
            insights=experiment.insights,
            warnings=[],
            total_duration_seconds=experiment.total_duration_seconds,
            status=experiment.status,
            created_at=experiment.created_at,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Note: Import func at module level if using database counting
from sqlalchemy import func
