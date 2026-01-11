from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from ..db import Base


class UploadRecord(Base):
    __tablename__ = "upload_records"

    id = Column(Integer, primary_key=True, index=True)
    dataset_filename = Column(String, nullable=False)
    model_filename = Column(String, nullable=False)
    dataset_rows = Column(Integer)
    dataset_columns = Column(Integer)
    dataset_columns_list = Column(JSON)
    model_type = Column(String)
    model_supports_predict_proba = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())