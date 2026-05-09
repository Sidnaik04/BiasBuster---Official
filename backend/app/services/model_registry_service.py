"""Model Registry Service: Centralized ML model lifecycle management."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.models import ModelRegistry, UploadRecord
from app.schemas.model_registry import (
    RegisterModelRequest,
    ModelRegistryEntry,
    ModelComparisonResponse,
    ModelComparisonItem,
    ModelRecommendationResponse,
    ModelLineageResponse,
    ModelLineageNode,
    TradeoffAnalysis,
)
import uuid
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class ModelRegistryService:
    """Service for managing ML model lifecycle and governance."""

    @staticmethod
    async def register_model(
        payload: RegisterModelRequest,
        session: AsyncSession,
    ) -> tuple[bool, str, Optional[str]]:
        """
        Register a model in the registry.

        Args:
            payload: Model registration details
            session: Database session

        Returns:
            (success, message, model_id)
        """
        try:
            # Verify upload exists
            upload = (
                await session.execute(
                    select(UploadRecord).where(UploadRecord.id == payload.upload_id)
                )
            ).scalar_one_or_none()

            if not upload:
                return False, "Upload record not found", None

            # Verify artifact exists
            if not os.path.isfile(payload.artifact_path):
                return False, f"Artifact not found: {payload.artifact_path}", None

            # Generate model_id
            model_id = str(uuid.uuid4())

            # Compute combined score if not provided (should be provided, but safe default)
            combined_score = payload.combined_score
            fairness_score = payload.fairness_metrics.get("fairness_score", 0.5)
            accuracy = payload.performance_metrics.get("accuracy", 0.5)
            if combined_score == 0:
                combined_score = (0.6 * fairness_score) + (0.4 * accuracy)

            # Create registry entry
            entry = ModelRegistry(
                model_id=model_id,
                upload_id=payload.upload_id,
                model_name=payload.model_name,
                model_type=payload.model_type,
                source_type=payload.source_type,
                parent_model_id=payload.parent_model_id,
                optimization_method=payload.optimization_method,
                mitigation_strategy=payload.mitigation_strategy,
                retraining_method=payload.retraining_method,
                artifact_path=payload.artifact_path,
                artifact_size_bytes=payload.artifact_size_bytes,
                performance_metrics=payload.performance_metrics,
                fairness_metrics=payload.fairness_metrics,
                operational_metrics=payload.operational_metrics,
                combined_score=combined_score,
                version=payload.version,
                tags=payload.tags,
                notes=payload.notes,
                experiment_id=payload.experiment_id,
                parameters=payload.parameters,
            )

            session.add(entry)
            await session.commit()

            return True, f"Model {model_id} registered successfully", model_id

        except Exception as e:
            return False, f"Registration failed: {str(e)}", None

    @staticmethod
    async def get_model_by_id(
        model_id: str,
        session: AsyncSession,
    ) -> Optional[ModelRegistryEntry]:
        """Get a specific model from registry."""
        model = (
            await session.execute(
                select(ModelRegistry).where(ModelRegistry.model_id == model_id)
            )
        ).scalar_one_or_none()

        if not model:
            return None

        return ModelRegistryService._model_to_entry(model)

    @staticmethod
    async def get_models_by_upload(
        upload_id: int,
        session: AsyncSession,
    ) -> List[ModelRegistryEntry]:
        """Get all models for a specific upload."""
        models = (
            (
                await session.execute(
                    select(ModelRegistry)
                    .where(ModelRegistry.upload_id == upload_id)
                    .order_by(ModelRegistry.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        return [ModelRegistryService._model_to_entry(m) for m in models]

    @staticmethod
    async def compare_models(
        upload_id: int,
        session: AsyncSession,
    ) -> Optional[ModelComparisonResponse]:
        """Compare all models for an upload and provide insights."""
        models = await ModelRegistryService.get_models_by_upload(upload_id, session)

        if not models:
            return None

        # Build comparison items
        comparison_items: List[ModelComparisonItem] = []
        accuracy_scores = []
        fairness_scores = []
        combined_scores = []

        best_accuracy_model = None
        best_fairness_model = None
        best_balanced_model = None
        best_production_model = None

        for model in models:
            accuracy = model.performance_metrics.get("accuracy", 0)
            fairness = model.fairness_metrics.get("fairness_score", 0)

            accuracy_scores.append(accuracy)
            fairness_scores.append(fairness)
            combined_scores.append(model.combined_score)

            item = ModelComparisonItem(
                model_id=model.model_id,
                model_name=model.model_name,
                source_type=model.source_type,
                version=model.version,
                accuracy=accuracy,
                fairness_score=fairness,
                combined_score=model.combined_score,
                dpd=model.fairness_metrics.get("dpd"),
                eod=model.fairness_metrics.get("eod"),
                is_recommended=(model.recommended_for is not None),
            )
            comparison_items.append(item)

            # Track best models
            if (
                best_accuracy_model is None
                or accuracy > best_accuracy_model["accuracy"]
            ):
                best_accuracy_model = {
                    "model_id": model.model_id,
                    "accuracy": accuracy,
                }
            if (
                best_fairness_model is None
                or fairness > best_fairness_model["fairness"]
            ):
                best_fairness_model = {"model_id": model.model_id, "fairness": fairness}

            if (
                best_balanced_model is None
                or model.combined_score > best_balanced_model["score"]
            ):
                best_balanced_model = {
                    "model_id": model.model_id,
                    "score": model.combined_score,
                }

            # Production best = retrained or stable optimized
            if model.source_type in ["retrained", "corrected"]:
                if (
                    best_production_model is None
                    or model.combined_score > best_production_model["score"]
                ):
                    best_production_model = {
                        "model_id": model.model_id,
                        "score": model.combined_score,
                    }

        # Statistics
        statistics = {
            "accuracy": {
                "min": min(accuracy_scores) if accuracy_scores else 0,
                "max": max(accuracy_scores) if accuracy_scores else 0,
                "avg": (
                    sum(accuracy_scores) / len(accuracy_scores)
                    if accuracy_scores
                    else 0
                ),
            },
            "fairness_score": {
                "min": min(fairness_scores) if fairness_scores else 0,
                "max": max(fairness_scores) if fairness_scores else 0,
                "avg": (
                    sum(fairness_scores) / len(fairness_scores)
                    if fairness_scores
                    else 0
                ),
            },
            "combined_score": {
                "min": min(combined_scores) if combined_scores else 0,
                "max": max(combined_scores) if combined_scores else 0,
                "avg": (
                    sum(combined_scores) / len(combined_scores)
                    if combined_scores
                    else 0
                ),
            },
            "total_models": len(models),
        }

        # Generate summary
        summary = ModelRegistryService._generate_comparison_summary(
            models, best_balanced_model
        )

        return ModelComparisonResponse(
            upload_id=upload_id,
            models=comparison_items,
            statistics=statistics,
            best_accuracy_model=(
                best_accuracy_model["model_id"] if best_accuracy_model else None
            ),
            best_fairness_model=(
                best_fairness_model["model_id"] if best_fairness_model else None
            ),
            best_balanced_model=(
                best_balanced_model["model_id"] if best_balanced_model else None
            ),
            best_production_model=(
                best_production_model["model_id"] if best_production_model else None
            ),
            summary=summary,
        )

    @staticmethod
    async def recommend_model(
        upload_id: int,
        goal: str = "balanced",
        session: AsyncSession = None,
    ) -> Optional[ModelRecommendationResponse]:
        """
        Recommend best model based on user goal.

        Goals:
        - accuracy: highest accuracy model
        - fairness: highest fairness_score model
        - balanced: best combined_score
        - production: stable retrained/corrected model
        """
        models = await ModelRegistryService.get_models_by_upload(upload_id, session)

        if not models:
            return None

        recommended_model = None
        recommendation_reason = ""

        if goal == "accuracy":
            recommended_model = max(
                models,
                key=lambda m: m.performance_metrics.get("accuracy", 0),
            )
            recommendation_reason = (
                f"This model has the highest accuracy ({recommended_model.performance_metrics['accuracy']:.2%})"
                f" among all models for this dataset."
            )

        elif goal == "fairness":
            recommended_model = max(
                models,
                key=lambda m: m.fairness_metrics.get("fairness_score", 0),
            )
            recommendation_reason = (
                f"This model has the best fairness score ({recommended_model.fairness_metrics['fairness_score']:.2%})"
                f" with DPD: {recommended_model.fairness_metrics.get('dpd', 'N/A'):.4f}."
            )

        elif goal == "production":
            # Prefer retrained/corrected models, fallback to optimized
            prod_models = [
                m for m in models if m.source_type in ["retrained", "corrected"]
            ]
            if not prod_models:
                prod_models = [m for m in models if m.source_type == "optimized"]
            if not prod_models:
                prod_models = models

            recommended_model = max(
                prod_models,
                key=lambda m: m.combined_score,
            )
            recommendation_reason = (
                f"This {recommended_model.source_type} model offers the best production stability"
                f" with combined score: {recommended_model.combined_score:.4f}."
            )

        else:  # balanced (default)
            recommended_model = max(models, key=lambda m: m.combined_score)
            recommendation_reason = (
                f"This model achieves the best accuracy-fairness balance"
                f" (combined score: {recommended_model.combined_score:.4f})."
            )

        # Find alternatives
        alternatives = [m for m in models if m.model_id != recommended_model.model_id][
            :3
        ]

        # Generate tradeoff analysis vs next best
        tradeoffs: List[TradeoffAnalysis] = []
        if len(models) > 1:
            other_models = [
                m for m in models if m.model_id != recommended_model.model_id
            ]
            for other in other_models[:2]:
                acc_diff = recommended_model.performance_metrics.get(
                    "accuracy", 0
                ) - other.performance_metrics.get("accuracy", 0)
                fair_diff = recommended_model.fairness_metrics.get(
                    "fairness_score", 0
                ) - other.fairness_metrics.get("fairness_score", 0)
                combined_diff = recommended_model.combined_score - other.combined_score

                if acc_diff > 0 and fair_diff > 0:
                    explanation = (
                        f"Recommended model is superior in both accuracy "
                        f"(+{abs(acc_diff):.2%}) and fairness (+{abs(fair_diff):.2%})."
                    )
                elif acc_diff > 0:
                    explanation = (
                        f"Recommended model trades +{abs(acc_diff):.2%} accuracy "
                        f"for -{abs(fair_diff):.2%} fairness."
                    )
                else:
                    explanation = (
                        f"Recommended model trades -{abs(acc_diff):.2%} accuracy "
                        f"for +{abs(fair_diff):.2%} fairness improvement."
                    )

                tradeoffs.append(
                    TradeoffAnalysis(
                        model_pair={
                            recommended_model.model_id: recommended_model.model_name,
                            other.model_id: other.model_name,
                        },
                        accuracy_difference=round(acc_diff, 4),
                        fairness_difference=round(fair_diff, 4),
                        combined_score_difference=round(combined_diff, 4),
                        recommendation=(
                            "Recommended" if combined_diff > 0 else "Alternative"
                        ),
                        explanation=explanation,
                    )
                )

        return ModelRecommendationResponse(
            recommended_model_id=recommended_model.model_id,
            recommended_model=recommended_model,
            goal=goal,
            reasoning=recommendation_reason,
            tradeoff_analysis=tradeoffs if tradeoffs else None,
            alternatives=alternatives,
        )

    @staticmethod
    async def get_lineage(
        model_id: str,
        session: AsyncSession,
    ) -> Optional[ModelLineageResponse]:
        """Get full lineage tree for a model and its descendants."""
        # Find root (model with no parent)
        model = await ModelRegistryService.get_model_by_id(model_id, session)
        if not model:
            return None

        # Trace back to root
        current = model
        while current.parent_model_id:
            current = await ModelRegistryService.get_model_by_id(
                current.parent_model_id, session
            )
            if not current:
                break

        # Build tree from root
        all_models = await ModelRegistryService.get_models_by_upload(
            model.upload_id, session
        )
        tree = await ModelRegistryService._build_lineage_tree(
            current.model_id if current else model_id, all_models, session
        )

        if not tree:
            return None

        def count_depth(node: ModelLineageNode) -> int:
            if not node.children:
                return 1
            return 1 + max(count_depth(child) for child in node.children)

        return ModelLineageResponse(
            root=tree,
            all_models=all_models,
            depth=count_depth(tree),
        )

    @staticmethod
    async def toggle_favorite(
        model_id: str,
        session: AsyncSession,
    ) -> tuple[bool, str]:
        """Toggle favorite status of a model."""
        try:
            model = (
                await session.execute(
                    select(ModelRegistry).where(ModelRegistry.model_id == model_id)
                )
            ).scalar_one_or_none()

            if not model:
                return False, "Model not found"

            model.is_favorite = not model.is_favorite
            session.add(model)
            await session.commit()

            return True, f"Model favorite status: {model.is_favorite}"

        except Exception as e:
            return False, f"Error: {str(e)}"

    @staticmethod
    async def add_model_notes(
        model_id: str,
        notes: str,
        session: AsyncSession,
    ) -> tuple[bool, str]:
        """Add or update notes for a model."""
        try:
            model = (
                await session.execute(
                    select(ModelRegistry).where(ModelRegistry.model_id == model_id)
                )
            ).scalar_one_or_none()

            if not model:
                return False, "Model not found"

            model.notes = notes
            session.add(model)
            await session.commit()

            return True, "Notes updated"

        except Exception as e:
            return False, f"Error: {str(e)}"

    # -- Helper methods --

    @staticmethod
    def _model_to_entry(model: ModelRegistry) -> ModelRegistryEntry:
        """Convert DB model to Pydantic schema."""
        return ModelRegistryEntry(
            model_id=model.model_id,
            upload_id=model.upload_id,
            model_name=model.model_name,
            model_type=model.model_type,
            source_type=model.source_type,
            parent_model_id=model.parent_model_id,
            optimization_method=model.optimization_method,
            mitigation_strategy=model.mitigation_strategy,
            retraining_method=model.retraining_method,
            artifact_path=model.artifact_path,
            artifact_size_bytes=model.artifact_size_bytes,
            performance_metrics=model.performance_metrics,
            fairness_metrics=model.fairness_metrics,
            operational_metrics=model.operational_metrics,
            combined_score=model.combined_score,
            recommended_for=model.recommended_for,
            recommendation_reason=model.recommendation_reason,
            version=model.version,
            tags=model.tags,
            notes=model.notes,
            experiment_id=model.experiment_id,
            parameters=model.parameters,
            is_favorite=model.is_favorite,
            is_available=model.is_available,
            created_at=model.created_at.isoformat() if model.created_at else "",
        )

    @staticmethod
    def _generate_comparison_summary(
        models: List[ModelRegistryEntry],
        best_balanced_model: Optional[Dict[str, Any]],
    ) -> str:
        """Generate human-readable summary of model comparison."""
        if not models:
            return "No models to compare."

        if len(models) == 1:
            model = models[0]
            return (
                f"Single {model.source_type} model with accuracy "
                f"{model.performance_metrics['accuracy']:.2%} and fairness "
                f"{model.fairness_metrics['fairness_score']:.2%}."
            )

        varied_types = len(set(m.source_type for m in models)) > 1
        if varied_types:
            return (
                f"Registry contains {len(models)} models across multiple transformation types. "
                f"Best balanced model ({best_balanced_model.get('model_id', 'N/A')}) "
                f"achieved the optimal accuracy-fairness tradeoff."
            )
        else:
            source = models[0].source_type
            return (
                f"{len(models)} {source} model variants. "
                f"Best variant achieves combined score "
                f"{best_balanced_model.get('score', 0):.4f}."
            )

    @staticmethod
    async def _build_lineage_tree(
        root_id: str,
        all_models: List[ModelRegistryEntry],
        session: AsyncSession,
    ) -> Optional[ModelLineageNode]:
        """Build lineage tree recursively."""
        root = next((m for m in all_models if m.model_id == root_id), None)
        if not root:
            return None

        children = [m for m in all_models if m.parent_model_id == root_id]

        child_nodes = []
        for child in children:
            child_node = await ModelRegistryService._build_lineage_tree(
                child.model_id, all_models, session
            )
            if child_node:
                child_nodes.append(child_node)

        return ModelLineageNode(
            model_id=root.model_id,
            model_name=root.model_name,
            source_type=root.source_type,
            version=root.version,
            combined_score=root.combined_score,
            children=child_nodes,
        )
