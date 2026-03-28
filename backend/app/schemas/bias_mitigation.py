from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class BiasMitigationRequest(BaseModel):
    upload_id: int
    target_column: str
    sensitive_attributes: List[str]
    bias_ranking: List[str]
    bias_scores: Dict[str, float]

    chosen_strategy: str = Field(..., description="smote | reweighting | threshold")

    strategy_config: Optional[Dict] = Field(
        default_factory=dict,
        description="User overrides (threshold value, split ratio, etc.)",
    )

    confirm_recommendation: bool = Field(
        ..., description="Human-in-the-loop confirmation"
    )
