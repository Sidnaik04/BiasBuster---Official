from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# =====================================================
# REQUEST SCHEMAS
# =====================================================


class ExperimentRunRequest(BaseModel):
    """Request to run experimentation engine."""

    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes to test"
    )
    strategies: List[str] = Field(
        default=["threshold", "reweighting", "smote"],
        description="Strategies to test (default: all)",
    )
    timeout_per_strategy: int = Field(
        default=300, description="Timeout per strategy in seconds"
    )


# =====================================================
# RESPONSE SCHEMAS
# =====================================================


class StrategyResult(BaseModel):
    """Result of a single strategy."""

    strategy: str = Field(..., description="Strategy name")
    status: str = Field(default="success", description="success or failed")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Before metrics
    accuracy_before: Optional[float] = Field(None, description="Baseline accuracy")
    dpd_before: Optional[float] = Field(None, description="Baseline DPD")
    eod_before: Optional[float] = Field(None, description="Baseline EOD")
    dir_before: Optional[float] = Field(None, description="Baseline DIR")
    fairness_score_before: Optional[float] = Field(
        None, description="Baseline fairness score"
    )

    # After metrics
    accuracy_after: Optional[float] = Field(
        None, description="Accuracy after mitigation"
    )
    dpd_after: Optional[float] = Field(None, description="DPD after mitigation")
    eod_after: Optional[float] = Field(None, description="EOD after mitigation")
    dir_after: Optional[float] = Field(None, description="DIR after mitigation")
    fairness_score_after: Optional[float] = Field(
        None, description="Fairness score after mitigation"
    )

    # Improvements
    accuracy_drop: Optional[float] = Field(
        None, description="Change in accuracy (negative = drop)"
    )
    fairness_improvement: Optional[float] = Field(
        None, description="Improvement in fairness score"
    )
    combined_score: Optional[float] = Field(
        None, description="0.6*fairness_imp - 0.4*accuracy_drop"
    )

    # Metadata
    duration_seconds: Optional[float] = Field(None, description="Runtime in seconds")


class ExperimentReportResponse(BaseModel):
    """Final experiment report."""

    experiment_id: str = Field(..., description="Unique experiment ID")
    upload_id: int = Field(..., description="Original upload ID")
    target_column: str = Field(..., description="Target variable")
    sensitive_columns: List[str] = Field(..., description="Sensitive attributes tested")

    # Baseline
    metrics_before: Dict[str, Any] = Field(..., description="Baseline metrics")

    # Results
    strategies_tested: List[str] = Field(..., description="Strategies that were tested")
    results: List[StrategyResult] = Field(..., description="Results for each strategy")

    # Best strategy
    best_strategy: Optional[str] = Field(None, description="Best performing strategy")
    best_strategy_score: Optional[float] = Field(
        None, description="Score of best strategy"
    )

    # Insights
    insights: Optional[str] = Field(None, description="Generated explanation")
    warnings: List[str] = Field(default=[], description="Any warnings during run")

    # Metadata
    total_duration_seconds: Optional[float] = Field(None, description="Total runtime")
    status: str = Field(default="completed", description="completed or failed")
    created_at: datetime = Field(..., description="Creation timestamp")


class ExperimentHistoryResponse(BaseModel):
    """Single experiment in history."""

    experiment_id: str
    upload_id: int
    target_column: str
    best_strategy: Optional[str]
    best_strategy_score: Optional[float]
    created_at: datetime
    status: str


class ExperimentHistoryListResponse(BaseModel):
    """List of experiments."""

    total: int = Field(..., description="Total number of experiments")
    experiments: List[ExperimentHistoryResponse] = Field(
        ..., description="List of experiments"
    )
