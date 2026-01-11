from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]

class ModelInfo(BaseModel):
    model_type: str
    supports_predict_proba: bool

class UploadSuccess(BaseModel):
    status: str = Field("success")
    dataset_info: DatasetInfo
    model_info: ModelInfo
    next_step: str = Field("select_sensitive_attribute")

class ErrorResponse(BaseModel):
    status: str = Field("error")
    error: str
    details: Optional[str] = None

class UploadRecordOut(BaseModel):
    id: int
    dataset_filename: str
    model_filename: str
    dataset_rows: Optional[int]
    dataset_columns: Optional[int]
    dataset_columns_list: Optional[List[str]]
    model_type: Optional[str]
    model_supports_predict_proba: Optional[bool]
    created_at: datetime

    model_config = {"from_attributes": True}
