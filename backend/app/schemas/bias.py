from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class BiasDetectRequest(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes selected by user"
    )


class StrategyRecommendationRequest(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes selected by user"
    )


class ExplainBiasRequest(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes for bias explanation"
    )


class BiasContributor(BaseModel):
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Normalized importance score (0-1)")
    reason: str = Field(..., description="Human-readable explanation of contribution")


class GroupAnalysis(BaseModel):
    sensitive_attribute: str = Field(..., description="Name of sensitive attribute")
    group: str = Field(..., description="Group value")
    group_size: int = Field(..., description="Number of samples in group")
    selection_rate: float = Field(..., description="Rate of positive predictions")
    true_positive_rate: float = Field(..., description="TPR for the group")
    false_positive_rate: float = Field(..., description="FPR for the group")
    positive_prediction_count: int = Field(
        ..., description="Count of positive predictions"
    )
    observations: str = Field(..., description="Human-readable observations")


class ProxyFeature(BaseModel):
    feature: str = Field(..., description="Feature that acts as proxy")
    sensitive_attribute: str = Field(
        ..., description="Sensitive attribute it correlates with"
    )
    correlated_with: str = Field(..., description="What it's correlated with")
    correlation_score: float = Field(
        ..., ge=0.7, le=1.0, description="Correlation strength"
    )
    correlation_type: str = Field(
        ..., description="Type of correlation (Pearson or Encoded)"
    )
    reason: str = Field(..., description="Why this is considered a proxy")


class BiasExplanation(BaseModel):
    bias_detected: bool = Field(..., description="Whether bias was detected")
    top_bias_contributors: List[BiasContributor] = Field(
        ..., description="Top features contributing to bias"
    )
    group_analysis: List[GroupAnalysis] = Field(
        ..., description="Fairness metrics per group"
    )
    proxy_features: List[ProxyFeature] = Field(
        ..., description="Detected proxy/discriminatory features"
    )
    summary: str = Field(..., description="Human-readable summary of findings")


class CorrectBiasRequest(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes to address"
    )
    strategy_ids: List[int] = Field(
        ..., description="IDs of mitigation strategies to apply"
    )


class CorrectionResult(BaseModel):
    strategy_id: int = Field(..., description="Mitigation strategy ID")
    strategy_name: str = Field(..., description="Name of the strategy")
    original_fairness_score: float = Field(
        ..., description="Fairness score before correction"
    )
    corrected_fairness_score: float = Field(
        ..., description="Fairness score after correction"
    )
    improvement: float = Field(..., description="Percentage improvement")
    metrics_before: Dict[str, Any] = Field(
        ..., description="Fairness metrics before correction"
    )
    metrics_after: Dict[str, Any] = Field(
        ..., description="Fairness metrics after correction"
    )
    recommendation: str = Field(..., description="Whether to apply this strategy")


class CorrectBiasResponse(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    correction_results: List[CorrectionResult] = Field(
        ..., description="Results for each applied strategy"
    )
    best_strategy: int = Field(..., description="ID of the best performing strategy")
    overall_summary: str = Field(
        ..., description="Overall summary of correction results"
    )
