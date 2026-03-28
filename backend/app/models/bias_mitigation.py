from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db import Base


class BiasMitigationRun(Base):
    __tablename__ = "bias_mitigation_runs"

    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, ForeignKey("upload_records.id"))
    sensitive_attribute = Column(String)
    strategy_used = Column(String)
    config = Column(JSON)
    artifact_model_path = Column(String)
    artifact_dataset_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
