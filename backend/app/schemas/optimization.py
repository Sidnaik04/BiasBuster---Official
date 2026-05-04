from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OptimizationRequest(BaseModel):
    """Request for model optimization."""

    upload_id: int = Field(..., description="ID of uploaded dataset and model")
    target_column: str = Field(..., description="Target column name")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attribute columns"
    )
    method: str = Field(
        default="optuna", description="Optimization method: 'gridsearch' or 'optuna'"
    )
    n_trials: int = Field(
        default=20, ge=5, le=100, description="Number of trials (for Optuna)"
    )
    cv_folds: int = Field(
        default=5, ge=2, le=10, description="Cross-validation folds (for GridSearch)"
    )
    accuracy_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight for accuracy in combined score"
    )
    fairness_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for fairness in combined score"
    )
    timeout: int = Field(
        default=300, ge=60, description="Timeout in seconds (for Optuna)"
    )


class TrialResult(BaseModel):
    """Individual trial result."""

    params: Dict[str, Any]
    accuracy: float
    fairness_score: float
    combined_score: float
    dpd: Optional[float] = None
    eod: Optional[float] = None


class OptimizationResponse(BaseModel):
    """Response from model optimization."""

    status: str = Field(..., description="Status: 'success' or 'error'")
    best_params: Dict[str, Any] = Field(..., description="Best hyperparameters found")
    best_score: float = Field(..., description="Best combined score achieved")
    accuracy: float = Field(..., description="Test accuracy of best model")
    fairness_score: float = Field(..., description="Fairness score of best model")
    combined_score: float = Field(..., description="Combined score of best model")
    dpd: Optional[float] = Field(None, description="Demographic parity difference")
    eod: Optional[float] = Field(None, description="Equal opportunity difference")
    optimization_method: str = Field(
        ..., description="Method used: 'gridsearch' or 'optuna'"
    )
    trials_run: int = Field(..., description="Number of trials executed")
    comparison: List[TrialResult] = Field(..., description="Top trials for comparison")
    accuracy_weight: float = Field(..., description="Weight used for accuracy")
    fairness_weight: float = Field(..., description="Weight used for fairness")
    message: str = Field(default="", description="Additional status message")
