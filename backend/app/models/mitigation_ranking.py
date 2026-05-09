from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, Text, ForeignKey
from sqlalchemy.sql import func

from app.db import Base


class MitigationRanking(Base):
    __tablename__ = "mitigation_rankings"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(
        Integer, ForeignKey("upload_records.id"), nullable=False, index=True
    )
    strategy = Column(String, nullable=False)
    metrics_before = Column(JSON, nullable=False)
    metrics_after = Column(JSON, nullable=False)
    ranking_score = Column(Float, nullable=False)
    rank_position = Column(Integer, nullable=False)
    status = Column(String, default="success", nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
