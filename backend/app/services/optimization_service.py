"""Optimization service for hyperparameter tuning."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.models import UploadRecord
from app.utils.dataset_loader import load_dataset
from app.utils.model_loader import load_model
from app.utils.optimization.model_optimizer import (
    FairnessAwareOptimizer,
    get_param_grid,
    get_optuna_params,
    get_model_class_name,
    PARAM_GRIDS,
    OPTUNA_PARAMS,
)
from app.utils.target_encoder import encode_target_column
from app.utils.sensitive_validation import validate_sensitive_columns
from app.utils.sensitive_preprocessing import bin_age_column
from app.schemas.optimization import OptimizationRequest, OptimizationResponse

import numpy as np
import pandas as pd
import uuid
import os
import joblib

from app.models.models import OptimizationRun
from app.utils.fairness.evaluation_engine import evaluate_model_fairness
from app.schemas.model_registry import RegisterModelRequest
from app.services.model_registry_service import ModelRegistryService
from sklearn.base import clone


def _normalize_params_for_model(model, params: dict) -> dict:
    """Normalize parameter names to match model.set_params expectations."""
    if not params:
        return {}

    valid_keys = set(model.get_params().keys())

    # Fast path: already valid for this model.
    if set(params.keys()).issubset(valid_keys):
        return params

    if hasattr(model, "steps") and model.steps:
        step_name = model.steps[-1][0]
        normalized = {}

        for key, value in params.items():
            if key in valid_keys:
                normalized[key] = value
                continue

            prefixed_key = f"{step_name}__{key}"
            if prefixed_key in valid_keys:
                normalized[prefixed_key] = value

        return normalized

    # Non-pipeline fallback: ignore unknown keys rather than failing hard.
    return {k: v for k, v in params.items() if k in valid_keys}


async def run_model_optimization(
    payload: OptimizationRequest,
    session: AsyncSession,
) -> OptimizationResponse:
    """
    Run hyperparameter optimization on uploaded model.

    Args:
        payload: Optimization request
        session: Database session

    Returns:
        OptimizationResponse with results
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
            return OptimizationResponse(
                status="error",
                message="Upload record not found",
                best_params={},
                best_score=0.0,
                optimization_method=payload.method,
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
            )

        # -------------------------------------------------
        # STEP 2: Load dataset & model
        # -------------------------------------------------
        df = load_dataset(record.dataset_filename)
        model = load_model(record.model_filename)

        # -------------------------------------------------
        # STEP 3: Target encoding
        # -------------------------------------------------
        df, target_info = encode_target_column(df, payload.target_column)

        # -------------------------------------------------
        # STEP 4: Sensitive attributes validation & preprocessing
        # -------------------------------------------------
        sensitive_info = validate_sensitive_columns(df, payload.sensitive_columns)

        # Keep request payload immutable for compatibility with Pydantic v2 models.
        original_sensitive_columns = list(payload.sensitive_columns)
        processed_sensitive_columns = list(payload.sensitive_columns)

        # Handle age binning
        for col in processed_sensitive_columns:
            if col.lower() == "age":
                df = bin_age_column(df, col)
                processed_sensitive_columns = [
                    c if c != col else col + "_group"
                    for c in processed_sensitive_columns
                ]
                break

        # Convert to string for consistency
        for col in processed_sensitive_columns:
            df[col] = df[col].astype(str)

        # -------------------------------------------------
        # STEP 5: Prepare features
        # -------------------------------------------------
        # NOTE: Keep sensitive attributes in X!
        # The model was trained with them, and removing them breaks Pipelines
        # Sensitive attributes are used separately for fairness metrics

        X = df[[col for col in df.columns if col != payload.target_column]]
        y = df[payload.target_column].values
        sensitive_features = df[processed_sensitive_columns]

        # Keep original dataframe for evaluate_model_fairness.
        df_original = load_dataset(record.dataset_filename)

        # -------------------------------------------------
        # STEP 6: Get model class name for parameter defaults
        # -------------------------------------------------
        model_class_name = get_model_class_name(model)

        # Verify weights sum to 1
        total_weight = payload.accuracy_weight + payload.fairness_weight
        if not (0.99 < total_weight < 1.01):  # Allow small floating point error
            return OptimizationResponse(
                status="error",
                message="accuracy_weight + fairness_weight must equal 1.0",
                best_params={},
                best_score=0.0,
                optimization_method=payload.method,
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
            )

        # -------------------------------------------------
        # STEP 6.5: Evaluate baseline
        # -------------------------------------------------
        baseline_eval = evaluate_model_fairness(
            df=df_original,
            model=model,
            target_column=payload.target_column,
            sensitive_columns=original_sensitive_columns,
        )
        baseline_accuracy = baseline_eval["performance"]["accuracy"]
        baseline_fairness = baseline_eval["fairness"]["aggregate"]["fairness_score"]
        baseline_combined = (payload.accuracy_weight * baseline_accuracy) + (
            payload.fairness_weight * baseline_fairness
        )

        # -------------------------------------------------
        # STEP 7: Create optimizer
        # -------------------------------------------------
        optimizer = FairnessAwareOptimizer(
            model=model,
            X=X,
            y=y,
            sensitive_features=sensitive_features,
            test_size=0.3,
            random_state=42,
            accuracy_weight=payload.accuracy_weight,
            fairness_weight=payload.fairness_weight,
        )

        # -------------------------------------------------
        # STEP 8: Run optimization
        # -------------------------------------------------
        if payload.method.lower() == "gridsearch":
            # Get parameter grid
            param_grid = get_param_grid(model_class_name)

            if not param_grid:
                return OptimizationResponse(
                    status="error",
                    message=f"No parameter grid available for {model_class_name}",
                    best_params={},
                    best_score=0.0,
                    optimization_method=payload.method,
                    accuracy_weight=payload.accuracy_weight,
                    fairness_weight=payload.fairness_weight,
                )

            result = optimizer.optimize_with_gridsearch(
                param_grid=param_grid, cv=payload.cv_folds
            )

        elif payload.method.lower() == "optuna":
            # Get parameter distributions
            param_distributions = get_optuna_params(model_class_name)

            if not param_distributions:
                return OptimizationResponse(
                    status="error",
                    message=f"No Optuna parameters available for {model_class_name}",
                    best_params={},
                    best_score=0.0,
                    optimization_method=payload.method,
                    accuracy_weight=payload.accuracy_weight,
                    fairness_weight=payload.fairness_weight,
                )

            result = optimizer.optimize_with_optuna(
                param_distributions=param_distributions,
                n_trials=payload.n_trials,
                timeout=payload.timeout,
            )

        else:
            return OptimizationResponse(
                status="error",
                message=f"Unknown optimization method: {payload.method}",
                best_params={},
                best_score=0.0,
                optimization_method=payload.method,
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
            )

        # -------------------------------------------------
        # STEP 9: Re-train with best parameters
        # -------------------------------------------------
        optimized_model = clone(model)
        best_params = result["best_params"]
        params_to_apply = _normalize_params_for_model(optimized_model, best_params)

        # Apply parameters
        optimized_model.set_params(**params_to_apply)

        optimized_model.fit(X, y)

        # -------------------------------------------------
        # STEP 10: Evaluate optimized model
        # -------------------------------------------------
        optimized_eval = evaluate_model_fairness(
            df=df_original,
            model=optimized_model,
            target_column=payload.target_column,
            sensitive_columns=original_sensitive_columns,
        )

        optimized_accuracy = optimized_eval["performance"]["accuracy"]
        optimized_fairness = optimized_eval["fairness"]["aggregate"]["fairness_score"]
        optimized_combined = (payload.accuracy_weight * optimized_accuracy) + (
            payload.fairness_weight * optimized_fairness
        )

        # -------------------------------------------------
        # STEP 11: Improvements & Export
        # -------------------------------------------------
        acc_gain = optimized_accuracy - baseline_accuracy
        fairness_gain = optimized_fairness - baseline_fairness
        f1_gain = (
            optimized_eval["performance"]["f1"] - baseline_eval["performance"]["f1"]
        )
        roc_auc_gain = (optimized_eval["performance"]["roc_auc"] or 0) - (
            baseline_eval["performance"]["roc_auc"] or 0
        )

        improvements = {
            "accuracy_gain_percent": round(acc_gain * 100, 2),
            "fairness_gain_percent": round(fairness_gain * 100, 2),
            "f1_gain_percent": round(f1_gain * 100, 2),
            "roc_auc_gain_percent": round(roc_auc_gain * 100, 2),
        }

        optimization_id = str(uuid.uuid4())

        optimized_model_available = False
        artifact_path = None
        download_endpoint = None

        if optimized_combined > baseline_combined:
            os.makedirs("artifacts/optimized_models", exist_ok=True)
            artifact_path = (
                f"artifacts/optimized_models/optimized_{optimization_id}.joblib"
            )
            joblib.dump(optimized_model, artifact_path)
            optimized_model_available = True
            download_endpoint = f"/api/optimize/download/{optimization_id}"
            optimization_summary = f"Optimization improved accuracy by {improvements['accuracy_gain_percent']}% and fairness by {improvements['fairness_gain_percent']}%. Model exported."
        else:
            optimization_summary = (
                "Optimization did not improve baseline model significantly."
            )

        # Save to DB
        new_run = OptimizationRun(
            optimization_id=optimization_id,
            upload_id=payload.upload_id,
            target_column=payload.target_column,
            sensitive_columns=original_sensitive_columns,
            optimization_method=payload.method,
            best_params=best_params,
            metrics_before=baseline_eval,
            metrics_after=optimized_eval,
            improvements=improvements,
            artifact_path=artifact_path,
            status="success",
        )
        session.add(new_run)
        await session.commit()

        # Auto-register optimized model in ModelRegistry
        if artifact_path and optimized_model_available:
            model_class_name = get_model_class_name(optimized_model)
            version_num = (
                len(
                    await session.execute(
                        select(OptimizationRun).where(
                            OptimizationRun.upload_id == payload.upload_id
                        )
                    )
                )
                .scalars()
                .all()
            )

            registry_payload = RegisterModelRequest(
                upload_id=payload.upload_id,
                model_name=f"{model_class_name}_optimized_{payload.method}",
                model_type=model_class_name,
                source_type="optimized",
                parent_model_id=None,
                optimization_method=payload.method,
                artifact_path=artifact_path,
                artifact_size_bytes=(
                    os.path.getsize(artifact_path)
                    if os.path.isfile(artifact_path)
                    else None
                ),
                performance_metrics=optimized_eval["performance"],
                fairness_metrics=optimized_eval["fairness"]["aggregate"],
                operational_metrics={
                    "model_size_bytes": (
                        os.path.getsize(artifact_path)
                        if os.path.isfile(artifact_path)
                        else None
                    ),
                },
                combined_score=optimized_combined,
                version=f"v{version_num}_optimized",
                parameters=best_params,
                experiment_id=optimization_id,
            )

            await ModelRegistryService.register_model(registry_payload, session)

        return OptimizationResponse(
            status="success",
            optimization_id=optimization_id,
            baseline_model=baseline_eval,
            optimized_model=optimized_eval,
            improvements=improvements,
            best_params=best_params,
            best_score=optimized_combined,
            optimization_method=payload.method,
            optimized_model_available=optimized_model_available,
            download_endpoint=download_endpoint,
            optimization_summary=optimization_summary,
            trials_run=result["trials_run"],
            comparison=[
                {
                    "params": trial.get("params", {}),
                    "accuracy": trial.get("accuracy", 0.0),
                    "fairness_score": trial.get("fairness_score", 0.0),
                    "combined_score": trial.get("combined_score", 0.0),
                    "dpd": trial.get("dpd"),
                    "eod": trial.get("eod"),
                }
                for trial in result.get("comparison", [])
            ],
            accuracy_weight=payload.accuracy_weight,
            fairness_weight=payload.fairness_weight,
            message="Optimization completed successfully.",
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Optimization error: {error_details}")

        return OptimizationResponse(
            status="error",
            message=f"Optimization failed: {str(e)}",
            best_params={},
            best_score=0.0,
            optimization_method=payload.method,
            accuracy_weight=payload.accuracy_weight,
            fairness_weight=payload.fairness_weight,
        )
