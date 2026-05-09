from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class BiasMitigationRequest(BaseModel):
    """
    Pipeline-aware bias mitigation request.

    All required metrics (bias ranking, fairness scores, etc.) are computed
    internally from the dataset and model using upload_id.
    """

    upload_id: int = Field(..., description="ID of the uploaded dataset/model record")
    target_column: str = Field(..., description="Target column for fairness evaluation")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes to protect"
    )
    strategy_name: str = Field(
        ..., description="Mitigation strategy: 'threshold', 'reweighting', or 'smote'"
    )
    strategy_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Optional strategy-specific parameters (e.g., grid_size for threshold)",
    )
    confirm_recommendation: bool = Field(
        ..., description="Human-in-the-loop confirmation for mitigation execution"
    )


class MitigationRankingRequest(BaseModel):
    upload_id: int
    target_column: str
    sensitive_columns: List[str]
