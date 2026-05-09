"""Model Registry API Endpoints: ML Lifecycle Management."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.model_registry import (
    RegisterModelRequest,
    RegisterModelResponse,
    ModelComparisonResponse,
    ModelRecommendationResponse,
    ModelLineageResponse,
    ModelRegistryEntry,
    ModelDownloadResponse,
)
from app.services.model_registry_service import ModelRegistryService
import os

router = APIRouter(prefix="/api/models", tags=["Model Registry"])


@router.post("/register", response_model=RegisterModelResponse)
async def register_model(
    payload: RegisterModelRequest,
    session: AsyncSession = Depends(get_session),
) -> RegisterModelResponse:
    """
    Register a model in the centralized registry.

    This endpoint automatically registers models from any source:
    - Original uploaded models
    - Optimized models (GridSearch, Optuna)
    - Mitigated models (Reweighting, SMOTE, Threshold)
    - Retrained models
    - Corrected models

    Args:
        payload: Model registration details including metrics, artifact path, etc.

    Returns:
        {
            "status": "success" | "error",
            "model_id": "uuid",
            "message": "..."
        }
    """
    success, message, model_id = await ModelRegistryService.register_model(
        payload, session
    )

    return RegisterModelResponse(
        status="success" if success else "error",
        model_id=model_id,
        message=message,
    )


@router.get("/model/{model_id}", response_model=ModelRegistryEntry)
async def get_model(
    model_id: str,
    session: AsyncSession = Depends(get_session),
) -> ModelRegistryEntry:
    """
    Retrieve a specific model from registry by ID.

    Args:
        model_id: Unique model identifier (UUID)

    Returns:
        Complete model metadata and metrics
    """
    model = await ModelRegistryService.get_model_by_id(model_id, session)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return model


@router.get("/upload/{upload_id}", response_model=list[ModelRegistryEntry])
async def get_models_for_upload(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[ModelRegistryEntry]:
    """
    Get all models registered for a specific upload/dataset.

    Args:
        upload_id: ID of the uploaded dataset/model

    Returns:
        List of all models for this upload
    """
    models = await ModelRegistryService.get_models_by_upload(upload_id, session)
    return models


@router.get("/compare/{upload_id}", response_model=ModelComparisonResponse)
async def compare_models(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
) -> ModelComparisonResponse:
    """
    Compare all models for a given upload and extract insights.

    Returns:
    - All models with comparable metrics
    - Best models by accuracy, fairness, and balance
    - Statistics (min/max/avg across metrics)
    - Generated summary

    Args:
        upload_id: ID of the upload

    Returns:
        {
            "models": [...],
            "best_accuracy_model": "model_id",
            "best_fairness_model": "model_id",
            "best_balanced_model": "model_id",
            "best_production_model": "model_id",
            "statistics": {...},
            "summary": "..."
        }
    """
    comparison = await ModelRegistryService.compare_models(upload_id, session)

    if not comparison:
        raise HTTPException(
            status_code=404,
            detail="No models found for this upload",
        )

    return comparison


@router.get("/recommend/{upload_id}", response_model=ModelRecommendationResponse)
async def recommend_model(
    upload_id: int,
    goal: str = Query(
        "balanced",
        description="Recommendation goal: accuracy, fairness, balanced, or production",
    ),
    session: AsyncSession = Depends(get_session),
) -> ModelRecommendationResponse:
    """
    Get AI-powered model recommendation based on user goals.

    Goals:
    - **accuracy**: Highest accuracy regardless of fairness
    - **fairness**: Highest fairness score regardless of accuracy
    - **balanced** (default): Best accuracy-fairness balance
    - **production**: Most stable (prefers retrained/corrected models)

    Returns:
    - Recommended model with full metadata
    - Reasoning and tradeoff analysis
    - Alternative recommendations
    - Comparison with next-best models

    Args:
        upload_id: ID of the upload
        goal: Recommendation goal (default: "balanced")

    Returns:
        Detailed recommendation with reasoning
    """
    if goal not in ["accuracy", "fairness", "balanced", "production"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid goal. Must be: accuracy, fairness, balanced, or production",
        )

    recommendation = await ModelRegistryService.recommend_model(
        upload_id, goal, session
    )

    if not recommendation:
        raise HTTPException(
            status_code=404,
            detail="No models found for recommendation",
        )

    return recommendation


@router.get("/lineage/{model_id}", response_model=ModelLineageResponse)
async def get_model_lineage(
    model_id: str,
    session: AsyncSession = Depends(get_session),
) -> ModelLineageResponse:
    """
    Get complete lineage/version history of a model tree.

    Shows the transformation journey:
    original → optimized → mitigated → retrained

    Returns:
    - Root model (original)
    - Full tree of descendants
    - All models in the lineage
    - Tree depth

    Args:
        model_id: Starting model ID

    Returns:
        Complete lineage tree with all relationships
    """
    lineage = await ModelRegistryService.get_lineage(model_id, session)

    if not lineage:
        raise HTTPException(status_code=404, detail="Model not found")

    return lineage


@router.post("/favorite/{model_id}")
async def toggle_favorite(
    model_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Mark a model as favorite for quick access.

    Args:
        model_id: Model to mark

    Returns:
        {"success": true, "message": "..."}
    """
    success, message = await ModelRegistryService.toggle_favorite(model_id, session)

    if not success:
        raise HTTPException(status_code=404, detail=message)

    return {"success": success, "message": message}


@router.post("/notes/{model_id}")
async def add_notes(
    model_id: str,
    notes: str = Query(..., description="Notes to add to this model"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Add or update notes for a model.

    Args:
        model_id: Model ID
        notes: Custom notes

    Returns:
        {"success": true, "message": "..."}
    """
    success, message = await ModelRegistryService.add_model_notes(
        model_id, notes, session
    )

    if not success:
        raise HTTPException(status_code=404, detail=message)

    return {"success": success, "message": message}


@router.get("/download/{model_id}")
async def download_model(
    model_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Download a model artifact as .joblib file.

    Args:
        model_id: Model to download

    Returns:
        Binary model file with proper headers
    """
    model = await ModelRegistryService.get_model_by_id(model_id, session)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    artifact_path = model.artifact_path

    if not os.path.isfile(artifact_path):
        raise HTTPException(
            status_code=404,
            detail=f"Artifact file not found: {artifact_path}",
        )

    return FileResponse(
        path=artifact_path,
        filename=f"{model.model_id}.joblib",
        media_type="application/octet-stream",
    )


@router.get("/summary/{upload_id}")
async def get_registry_summary(
    upload_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Get high-level summary of all models for an upload.

    Returns:
    - Total models
    - Source type distribution
    - Best metrics achieved
    - Recommended model

    Args:
        upload_id: Upload ID

    Returns:
        Compact summary for dashboard display
    """
    models = await ModelRegistryService.get_models_by_upload(upload_id, session)

    if not models:
        raise HTTPException(
            status_code=404,
            detail="No models found for this upload",
        )

    source_types = {}
    for model in models:
        source_types[model.source_type] = source_types.get(model.source_type, 0) + 1

    best_model = max(models, key=lambda m: m.combined_score)
    best_accuracy = max(models, key=lambda m: m.performance_metrics.get("accuracy", 0))
    best_fairness = max(
        models, key=lambda m: m.fairness_metrics.get("fairness_score", 0)
    )

    return {
        "upload_id": upload_id,
        "total_models": len(models),
        "source_types": source_types,
        "best_model": {
            "model_id": best_model.model_id,
            "model_name": best_model.model_name,
            "source_type": best_model.source_type,
            "combined_score": best_model.combined_score,
        },
        "best_accuracy_model": {
            "model_id": best_accuracy.model_id,
            "accuracy": best_accuracy.performance_metrics.get("accuracy", 0),
        },
        "best_fairness_model": {
            "model_id": best_fairness.model_id,
            "fairness_score": best_fairness.fairness_metrics.get("fairness_score", 0),
        },
        "first_registered": models[-1].created_at if models else None,
        "last_registered": models[0].created_at if models else None,
    }
