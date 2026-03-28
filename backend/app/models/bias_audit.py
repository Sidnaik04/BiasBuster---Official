from sqlalchemy import Column, Integer, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db import Base


class BiasAuditRecord(Base):
    __tablename__ = "bias_audit_records"

    id = Column(Integer, primary_key=True)
    upload_id = Column(Integer, ForeignKey("upload_records.id"))
    audit_stage = Column(Integer)  # 0 = before, 1 = after
    audit_result = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
