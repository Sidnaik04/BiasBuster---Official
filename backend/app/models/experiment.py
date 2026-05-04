from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey
from sqlalchemy.sql import func
from app.db import Base


class ExperimentRun(Base):
    """
    Stores experiment runs containing multiple mitigation strategy results.

    Fields:
        - experiment_id: Unique experiment identifier
        - upload_id: Reference to uploaded dataset/model
        - target_column: Target variable name
        - sensitive_columns: List of sensitive attributes
        - strategies_tested: List of strategies run (e.g., ['threshold', 'reweighting', 'smote'])
        - metrics_before: Baseline metrics (accuracy, dpd, eod, dir)
        - results: Array of strategy results
        - best_strategy: Best performing strategy
        - combined_score: Score used for best strategy selection
        - insights: Generated explanations
        - created_at: Timestamp
    """

    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True, nullable=False)
    upload_id = Column(Integer, ForeignKey("upload_records.id"), nullable=False)
    target_column = Column(String, nullable=False)
    sensitive_columns = Column(JSON, nullable=False)  # List of column names
    strategies_tested = Column(
        JSON, nullable=False
    )  # ['threshold', 'reweighting', 'smote']

    # Baseline metrics before any mitigation
    metrics_before = Column(
        JSON, nullable=False
    )  # {accuracy, dpd, eod, dir, fairness_score}

    # Results for each strategy
    results = Column(JSON, nullable=False)  # List of strategy results

    # Best strategy selection
    best_strategy = Column(String, nullable=True)
    combined_score = Column(Float, nullable=True)

    # Generated insights
    insights = Column(String, nullable=True)

    # Metadata
    total_duration_seconds = Column(Float, nullable=True)
    status = Column(String, default="completed")  # completed, failed, in_progress
    error_message = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
