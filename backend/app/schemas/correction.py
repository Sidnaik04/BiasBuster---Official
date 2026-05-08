from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class DataCorrectionWizardRequest(BaseModel):
    """
    Request to run the Data Correction Wizard.

    Applies fairness mitigation and generates debiased dataset/model.
    """

    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes to address"
    )
    strategy: str = Field(
        ..., description="Mitigation strategy: 'threshold', 'reweighting', or 'smote'"
    )
    export_format: str = Field(
        default="csv", description="Dataset export format: 'csv'"
    )


class CorrectionMetrics(BaseModel):
    """Fairness and performance metrics."""

    accuracy: float = Field(..., description="Model accuracy")
    dpd: float = Field(..., description="Demographic Parity Difference")
    eod: float = Field(..., description="Equalized Odds Difference")
    fairness_score: float = Field(
        ..., description="Fairness score (1 - avg(|DPD|, |EOD|))"
    )


class CorrectionImprovements(BaseModel):
    """Improvement metrics showing gains/losses."""

    fairness_gain: float = Field(..., description="Change in fairness score")
    accuracy_change: float = Field(..., description="Change in accuracy")
    dpd_reduction: float = Field(..., description="Reduction in DPD")
    eod_reduction: float = Field(..., description="Reduction in EOD")


class DataCorrectionWizardResponse(BaseModel):
    """
    Response from Data Correction Wizard.

    Contains corrected dataset, model paths, and before/after metrics.
    """

    correction_id: str = Field(..., description="Unique correction identifier")
    strategy_applied: str = Field(..., description="Applied mitigation strategy")
    dataset_export_path: str = Field(
        ..., description="Path to exported corrected dataset"
    )
    model_export_path: str = Field(
        ..., description="Path to exported corrected/retrained model"
    )
    report_export_path: str = Field(
        ..., description="Path to exported correction report (JSON metadata)"
    )
    metrics_before: CorrectionMetrics = Field(
        ..., description="Metrics before correction"
    )
    metrics_after: CorrectionMetrics = Field(
        ..., description="Metrics after correction"
    )
    improvements: CorrectionImprovements = Field(..., description="Improvement summary")
    summary: str = Field(
        ..., description="Human-readable summary of correction results"
    )


class CorrectionRecord(BaseModel):
    """Database record of a correction operation."""

    id: int = Field(..., description="Database ID")
    correction_id: str = Field(..., description="Unique correction identifier")
    upload_id: int = Field(..., description="Original upload ID")
    strategy: str = Field(..., description="Applied strategy")
    target_column: str = Field(..., description="Target column")
    sensitive_columns: List[str] = Field(..., description="Sensitive attributes")
    dataset_export_path: Optional[str] = Field(None, description="Dataset path")
    model_export_path: Optional[str] = Field(None, description="Model path")
    report_export_path: Optional[str] = Field(None, description="Report path")
    metrics_before: Dict[str, Any] = Field(..., description="Before metrics")
    metrics_after: Dict[str, Any] = Field(..., description="After metrics")
    improvements: Dict[str, Any] = Field(..., description="Improvements")
    summary: str = Field(..., description="Summary")
    status: str = Field(..., description="Operation status: success or failed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation timestamp")


class DownloadMetadata(BaseModel):
    """Metadata for downloadable artifacts."""

    correction_id: str = Field(..., description="Correction ID")
    strategy: str = Field(..., description="Strategy used")
    created_at: datetime = Field(..., description="Creation timestamp")
    file_type: str = Field(..., description="csv or joblib")
    file_path: str = Field(..., description="Path to file")
    file_size: int = Field(..., description="Size in bytes")
