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


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True)
    optimization_id = Column(String, unique=True, nullable=False, index=True)
    upload_id = Column(Integer, nullable=False)
    target_column = Column(String, nullable=False)
    sensitive_columns = Column(JSON, nullable=False)

    optimization_method = Column(String, nullable=False)
    best_params = Column(JSON, nullable=False)

    metrics_before = Column(JSON, nullable=False)
    metrics_after = Column(JSON, nullable=False)
    improvements = Column(JSON, nullable=False)

    artifact_path = Column(String, nullable=True)

    status = Column(String, default="success")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, nullable=False, index=True)  # UUID
    upload_id = Column(Integer, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    model_type = Column(
        String, nullable=False
    )  # e.g., LogisticRegression, RandomForest

    # source_type: original, optimized, mitigated, retrained, corrected
    source_type = Column(String, nullable=False, index=True)
    parent_model_id = Column(String, nullable=True, index=True)  # Lineage tracking

    # Optimization/Mitigation/Retraining details
    optimization_method = Column(String, nullable=True)  # gridsearch, optuna
    mitigation_strategy = Column(String, nullable=True)  # reweighting, smote, threshold
    retraining_method = Column(String, nullable=True)  # retrained_after_mitigation

    # Artifact management
    artifact_path = Column(String, nullable=False)
    artifact_size_bytes = Column(Integer, nullable=True)

    # Metrics stored as JSON
    performance_metrics = Column(
        JSON, nullable=False
    )  # accuracy, precision, recall, f1, roc_auc
    fairness_metrics = Column(JSON, nullable=False)  # dpd, eod, dir, fairness_score
    operational_metrics = Column(
        JSON, nullable=True
    )  # training_time, model_size, inference_latency

    # Combined score
    combined_score = Column(Float, nullable=False)  # 0.6*fairness + 0.4*accuracy

    # Recommendation metadata
    recommended_for = Column(
        String, nullable=True
    )  # best_accuracy, best_fairness, best_balanced, production
    recommendation_reason = Column(Text, nullable=True)

    # Versioning
    version = Column(String, nullable=False)  # v1_original, v2_optimized, etc.

    # Tags & metadata
    tags = Column(JSON, nullable=True)  # custom tags
    notes = Column(Text, nullable=True)

    # Tracking
    experiment_id = Column(String, nullable=True)
    parameters = Column(JSON, nullable=True)  # hyperparameters

    # Status
    is_favorite = Column(Boolean, default=False)
    is_available = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
