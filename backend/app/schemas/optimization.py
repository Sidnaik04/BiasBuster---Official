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
    optimization_id: Optional[str] = Field(None, description="Unique ID for this optimization run")
    baseline_model: Optional[Dict[str, Any]] = Field(None, description="Metrics for original model")
    optimized_model: Optional[Dict[str, Any]] = Field(None, description="Metrics for optimized model")
    improvements: Optional[Dict[str, Any]] = Field(None, description="Calculated improvements")
    best_params: Dict[str, Any] = Field(default_factory=dict, description="Best hyperparameters found")
    best_score: float = Field(default=0.0, description="Best combined score achieved")
    optimization_method: str = Field(..., description="Method used: 'gridsearch' or 'optuna'")
    optimized_model_available: bool = Field(default=False, description="Whether optimized model was saved")
    download_endpoint: Optional[str] = Field(None, description="Endpoint to download optimized model")
    optimization_summary: str = Field(default="", description="Summary of optimization results")
    
    # Original backward compatibility / trial details
    trials_run: int = Field(default=0, description="Number of trials executed")
    comparison: List[TrialResult] = Field(default_factory=list, description="Top trials for comparison")
    accuracy_weight: float = Field(..., description="Weight used for accuracy")
    fairness_weight: float = Field(..., description="Weight used for fairness")
    message: str = Field(default="", description="Additional status message")
