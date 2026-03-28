from pydantic import BaseModel, Field
from typing import List


class BiasDetectRequest(BaseModel):
    upload_id: int = Field(..., description="UploadRecord ID")
    target_column: str = Field(..., description="Target label column")
    sensitive_columns: List[str] = Field(
        ..., description="List of sensitive attributes selected by user"
    )
