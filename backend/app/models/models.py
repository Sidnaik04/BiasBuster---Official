from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Float, Text
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


class CorrectionRecord(Base):
    __tablename__ = "correction_records"

    id = Column(Integer, primary_key=True, index=True)
    correction_id = Column(String, unique=True, nullable=False, index=True)
    upload_id = Column(Integer, nullable=False)
    strategy = Column(String, nullable=False)  # threshold, reweighting, smote
    target_column = Column(String, nullable=False)
    sensitive_columns = Column(JSON, nullable=False)  # List of sensitive columns

    # Paths to exported artifacts
    dataset_export_path = Column(String, nullable=True)
    model_export_path = Column(String, nullable=True)
    report_export_path = Column(String, nullable=True)

    # Metrics before/after
    metrics_before = Column(
        JSON, nullable=False
    )  # Dict with accuracy, dpd, eod, fairness_score
    metrics_after = Column(JSON, nullable=False)

    # Improvements
    improvements = Column(
        JSON, nullable=False
    )  # fairness_gain, accuracy_change, dpd_reduction, eod_reduction

    # Summary
    summary = Column(Text, nullable=False)

    # Status
    status = Column(String, default="success")  # success, failed
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
