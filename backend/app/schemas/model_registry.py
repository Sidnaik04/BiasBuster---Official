from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ModelMetrics(BaseModel):
    """Standardized model metrics."""

    # Performance metrics
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None

    # Fairness metrics
    dpd: Optional[float] = None
    eod: Optional[float] = None
    dir: Optional[float] = None
    fairness_score: float


class ModelRegistryEntry(BaseModel):
    """Single model in registry."""

    model_id: str
    upload_id: int
    model_name: str
    model_type: str
    source_type: str  # original, optimized, mitigated, retrained, corrected
    parent_model_id: Optional[str] = None

    optimization_method: Optional[str] = None
    mitigation_strategy: Optional[str] = None
    retraining_method: Optional[str] = None

    artifact_path: str
    artifact_size_bytes: Optional[int] = None

    performance_metrics: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    operational_metrics: Optional[Dict[str, Any]] = None

    combined_score: float
    recommended_for: Optional[str] = None
    recommendation_reason: Optional[str] = None

    version: str
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    experiment_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    is_favorite: bool = False
    is_available: bool = True

    created_at: str


class ModelLineageNode(BaseModel):
    """Model in lineage tree."""

    model_id: str
    model_name: str
    source_type: str
    version: str
    combined_score: float
    children: List["ModelLineageNode"] = []


class ModelLineageResponse(BaseModel):
    """Full lineage tree for a model series."""

    root: ModelLineageNode
    all_models: List[ModelRegistryEntry]
    depth: int


class ModelComparisonItem(BaseModel):
    """Single model in comparison."""

    model_id: str
    model_name: str
    source_type: str
    version: str

    accuracy: float
    fairness_score: float
    combined_score: float

    dpd: Optional[float] = None
    eod: Optional[float] = None

    is_recommended: bool


class ModelComparisonResponse(BaseModel):
    """Comparison of all models for an upload."""

    upload_id: int
    models: List[ModelComparisonItem]

    statistics: Dict[str, Any] = Field(
        default_factory=dict, description="Min/max/avg metrics across all models"
    )

    best_accuracy_model: Optional[str] = None
    best_fairness_model: Optional[str] = None
    best_balanced_model: Optional[str] = None
    best_production_model: Optional[str] = None

    summary: str


class TradeoffAnalysis(BaseModel):
    """Tradeoff between accuracy and fairness."""

    model_pair: Dict[str, str]  # model_id -> name
    accuracy_difference: float
    fairness_difference: float
    combined_score_difference: float

    recommendation: str
    explanation: str


class ModelRecommendationResponse(BaseModel):
    """Model recommendation with explanation."""

    recommended_model_id: str
    recommended_model: ModelRegistryEntry

    goal: str  # accuracy, fairness, balanced, production

    reasoning: str
    tradeoff_analysis: Optional[List[TradeoffAnalysis]] = None

    alternatives: List[ModelRegistryEntry] = []


class RegisterModelRequest(BaseModel):
    """Request to register a new model."""

    upload_id: int
    model_name: str
    model_type: str
    source_type: str  # original, optimized, mitigated, retrained, corrected
    parent_model_id: Optional[str] = None

    optimization_method: Optional[str] = None
    mitigation_strategy: Optional[str] = None
    retraining_method: Optional[str] = None

    artifact_path: str
    artifact_size_bytes: Optional[int] = None

    performance_metrics: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    operational_metrics: Optional[Dict[str, Any]] = None

    combined_score: float

    version: str
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    experiment_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class RegisterModelResponse(BaseModel):
    """Response from model registration."""

    status: str  # success, error
    model_id: Optional[str] = None
    message: str


class ModelDownloadResponse(BaseModel):
    """Response for model download."""

    model_id: str
    model_name: str
    artifact_path: str
    download_url: str
