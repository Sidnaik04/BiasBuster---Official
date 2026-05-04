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
                best_params={},
                best_score=0.0,
                accuracy=0.0,
                fairness_score=0.0,
                combined_score=0.0,
                optimization_method=payload.method,
                trials_run=0,
                comparison=[],
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
                message="Upload record not found",
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

        # Handle age binning
        for col in payload.sensitive_columns:
            if col.lower() == "age":
                df = bin_age_column(df, col)
                payload.sensitive_columns = [
                    c if c != col else col + "_group" for c in payload.sensitive_columns
                ]
                break

        # Convert to string for consistency
        for col in payload.sensitive_columns:
            df[col] = df[col].astype(str)

        # -------------------------------------------------
        # STEP 5: Prepare features
        # -------------------------------------------------
        # NOTE: Keep sensitive attributes in X!
        # The model was trained with them, and removing them breaks Pipelines
        # Sensitive attributes are used separately for fairness metrics

        X = df[[col for col in df.columns if col != payload.target_column]]
        y = df[payload.target_column].values
        sensitive_features = df[payload.sensitive_columns]

        # -------------------------------------------------
        # STEP 6: Get model class name for parameter defaults
        # -------------------------------------------------
        model_class_name = get_model_class_name(model)

        # Verify weights sum to 1
        total_weight = payload.accuracy_weight + payload.fairness_weight
        if not (0.99 < total_weight < 1.01):  # Allow small floating point error
            return OptimizationResponse(
                status="error",
                best_params={},
                best_score=0.0,
                accuracy=0.0,
                fairness_score=0.0,
                combined_score=0.0,
                optimization_method=payload.method,
                trials_run=0,
                comparison=[],
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
                message="accuracy_weight + fairness_weight must equal 1.0",
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
                    best_params={},
                    best_score=0.0,
                    accuracy=0.0,
                    fairness_score=0.0,
                    combined_score=0.0,
                    optimization_method=payload.method,
                    trials_run=0,
                    comparison=[],
                    accuracy_weight=payload.accuracy_weight,
                    fairness_weight=payload.fairness_weight,
                    message=f"No parameter grid available for {model_class_name}",
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
                    best_params={},
                    best_score=0.0,
                    accuracy=0.0,
                    fairness_score=0.0,
                    combined_score=0.0,
                    optimization_method=payload.method,
                    trials_run=0,
                    comparison=[],
                    accuracy_weight=payload.accuracy_weight,
                    fairness_weight=payload.fairness_weight,
                    message=f"No Optuna parameters available for {model_class_name}",
                )

            result = optimizer.optimize_with_optuna(
                param_distributions=param_distributions,
                n_trials=payload.n_trials,
                timeout=payload.timeout,
            )

        else:
            return OptimizationResponse(
                status="error",
                best_params={},
                best_score=0.0,
                accuracy=0.0,
                fairness_score=0.0,
                combined_score=0.0,
                optimization_method=payload.method,
                trials_run=0,
                comparison=[],
                accuracy_weight=payload.accuracy_weight,
                fairness_weight=payload.fairness_weight,
                message=f"Unknown optimization method: {payload.method}",
            )

        # -------------------------------------------------
        # STEP 9: Build response
        # -------------------------------------------------
        return OptimizationResponse(
            status="success",
            best_params=result["best_params"],
            best_score=result["best_score"],
            accuracy=result.get("accuracy", 0.0),
            fairness_score=result.get("fairness_score", 0.0),
            combined_score=result.get("combined_score", 0.0),
            dpd=result.get("dpd"),
            eod=result.get("eod"),
            optimization_method=result["optimization_method"],
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
            message=f"Optimization completed with {result['optimization_method']} method",
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(f"Optimization error: {error_details}")

        return OptimizationResponse(
            status="error",
            best_params={},
            best_score=0.0,
            accuracy=0.0,
            fairness_score=0.0,
            combined_score=0.0,
            optimization_method=payload.method,
            trials_run=0,
            comparison=[],
            accuracy_weight=payload.accuracy_weight,
            fairness_weight=payload.fairness_weight,
            message=f"Optimization failed: {str(e)}",
        )
