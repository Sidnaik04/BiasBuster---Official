"""Mitigation orchestration service."""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from app.utils.mitigation.base_strategy import BaseMitigationStrategy
from app.utils.mitigation.strategy_threshold import ThresholdStrategy
from app.utils.mitigation.strategy_reweighting import ReweightingStrategy
from app.utils.mitigation.strategy_smote import SMOTEStrategy

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    """Result object for mitigation execution."""

    strategy_name: str
    status: str  # "success", "error", "warning"
    error_message: Optional[str] = None
    mitigated_model: Optional[Any] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    fairness_improvement: Optional[float] = None
    accuracy_impact: Optional[float] = None
    combined_score: Optional[float] = None
    execution_diagnostics: Dict[str, Any] = field(default_factory=dict)
    execution_time_seconds: Optional[float] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class MitigationOrchestrator:
    """Orchestrates unified mitigation strategy execution."""

    STRATEGY_CLASSES = {
        "threshold": ThresholdStrategy,
        "reweighting": ReweightingStrategy,
        "smote": SMOTEStrategy,
    }

    def __init__(self):
        """Initialize orchestrator."""
        self.last_result: Optional[MitigationResult] = None

    @staticmethod
    def get_available_strategies() -> list:
        """Get list of available strategies."""
        return list(MitigationOrchestrator.STRATEGY_CLASSES.keys())

    def run_strategy(
        self,
        strategy_name: str,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
        target_column: str = "target",
        compute_metrics_func=None,
        **strategy_kwargs,
    ) -> MitigationResult:
        """
        Execute a single mitigation strategy with unified orchestration.

        Args:
            strategy_name: Name of the strategy ("threshold", "reweighting", "smote")
            model: Base sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
            compute_metrics_func: Optional function to compute fairness/accuracy metrics
            **strategy_kwargs: Additional arguments for the specific strategy

        Returns:
            MitigationResult with execution status and metrics
        """
        import time

        start_time = time.time()
        result = MitigationResult(strategy_name=strategy_name, status="running")

        try:
            # Get strategy class
            if strategy_name not in self.STRATEGY_CLASSES:
                result.status = "error"
                result.error_message = f"Unknown strategy: {strategy_name}"
                logger.error(result.error_message)
                self.last_result = result
                return result

            StrategyClass = self.STRATEGY_CLASSES[strategy_name]

            # Instantiate strategy
            logger.info(f"Initializing {strategy_name} strategy...")
            strategy = StrategyClass(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sensitive_features_train=sensitive_features_train,
                sensitive_features_test=sensitive_features_test,
                target_column=target_column,
                **strategy_kwargs,
            )

            # Compute metrics before mitigation
            if compute_metrics_func:
                logger.info("Computing metrics before mitigation...")
                result.metrics_before = compute_metrics_func(
                    model, X_test, y_test, sensitive_features_test
                )

            # Fit strategy
            logger.info(f"Fitting {strategy_name} strategy...")
            fit_success = strategy.fit()

            if not fit_success:
                result.status = "error"
                result.error_message = strategy.diagnostics.get(
                    "error", f"{strategy_name} fit failed"
                )
                logger.error(result.error_message)
                result.execution_diagnostics = strategy.diagnostics
                self.last_result = result
                return result

            # Get mitigated model
            result.mitigated_model = strategy.mitigated_model

            # Make predictions on test set
            logger.info(f"Generating predictions with {strategy_name} model...")
            if strategy._requires_sensitive_features_for_prediction():
                test_predictions = strategy._predict_with_sensitive_features(X_test)
            else:
                test_predictions = strategy.predict(X_test)

            # Compute metrics after mitigation
            if compute_metrics_func:
                logger.info("Computing metrics after mitigation...")
                result.metrics_after = compute_metrics_func(
                    strategy.mitigated_model, X_test, y_test, sensitive_features_test
                )

                # Calculate improvements
                result.fairness_improvement = self._calculate_fairness_improvement(
                    result.metrics_before, result.metrics_after
                )
                result.accuracy_impact = self._calculate_accuracy_impact(
                    result.metrics_before, result.metrics_after
                )
                result.combined_score = self._calculate_combined_score(
                    result.metrics_after
                )

            result.execution_diagnostics = strategy.get_execution_diagnostics()
            result.status = "success"
            logger.info(f"{strategy_name} strategy executed successfully")

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            logger.error(
                f"{strategy_name} strategy execution failed: {result.error_message}"
            )

        finally:
            result.execution_time_seconds = time.time() - start_time
            self.last_result = result

        return result

    def run_multi_strategy_experiment(
        self,
        strategies: list,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
        target_column: str = "target",
        compute_metrics_func=None,
    ) -> Dict[str, MitigationResult]:
        """
        Execute multiple mitigation strategies and compare results.

        Args:
            strategies: List of strategy names to execute
            model: Base sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
            compute_metrics_func: Function to compute fairness/accuracy metrics

        Returns:
            Dictionary mapping strategy names to MitigationResult objects
        """
        results = {}

        logger.info(
            f"Starting multi-strategy experiment with {len(strategies)} strategies"
        )

        for strategy_name in strategies:
            logger.info(f"Executing strategy: {strategy_name}")
            result = self.run_strategy(
                strategy_name=strategy_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sensitive_features_train=sensitive_features_train,
                sensitive_features_test=sensitive_features_test,
                target_column=target_column,
                compute_metrics_func=compute_metrics_func,
            )
            results[strategy_name] = result

            if result.status == "error":
                logger.warning(
                    f"Strategy {strategy_name} failed: {result.error_message}"
                )

        logger.info("Multi-strategy experiment completed")
        return results

    @staticmethod
    def _calculate_fairness_improvement(
        metrics_before: Dict[str, float], metrics_after: Dict[str, float]
    ) -> float:
        """Calculate fairness improvement (reduction in bias metrics)."""
        if not metrics_before or not metrics_after:
            return 0.0

        # Fairness metrics to consider (lower is better)
        fairness_metrics = ["dpd", "eod", "dir"]

        improvements = []
        for metric in fairness_metrics:
            before = metrics_before.get(metric)
            after = metrics_after.get(metric)
            if before is not None and after is not None:
                improvement = abs(before) - abs(after)
                improvements.append(improvement)

        return float(np.mean(improvements)) if improvements else 0.0

    @staticmethod
    def _calculate_accuracy_impact(
        metrics_before: Dict[str, float], metrics_after: Dict[str, float]
    ) -> float:
        """Calculate accuracy impact (negative if accuracy decreased)."""
        if not metrics_before or not metrics_after:
            return 0.0

        before_acc = metrics_before.get("accuracy", 0)
        after_acc = metrics_after.get("accuracy", 0)

        return float(after_acc - before_acc)

    @staticmethod
    def _calculate_combined_score(metrics: Dict[str, float]) -> float:
        """
        Calculate combined fairness+accuracy score.

        Formula: 0.6 * fairness_score + 0.4 * accuracy_score
        where fairness_score = 1 - avg(|dpd|, |eod|, |dir|)
        """
        if not metrics:
            return 0.0

        # Calculate fairness component (lower bias is better)
        fairness_metrics = []
        for metric in ["dpd", "eod", "dir"]:
            val = metrics.get(metric)
            if val is not None:
                fairness_metrics.append(abs(val))

        fairness_score = 1.0 - np.mean(fairness_metrics) if fairness_metrics else 0.5

        # Calculate accuracy component
        accuracy = metrics.get("accuracy", 0.5)

        # Combined score
        combined = 0.6 * fairness_score + 0.4 * accuracy
        return float(np.clip(combined, 0, 1))

    def get_best_strategy(
        self, results: Dict[str, MitigationResult], metric: str = "combined_score"
    ) -> Tuple[str, MitigationResult]:
        """
        Recommend best strategy based on metric.

        Args:
            results: Dictionary of strategy results
            metric: Metric to optimize ("combined_score", "fairness_improvement", "accuracy")

        Returns:
            Tuple of (strategy_name, best_result)
        """
        successful_results = {
            name: result
            for name, result in results.items()
            if result.status == "success"
        }

        if not successful_results:
            return None, None

        if metric == "combined_score":
            best = max(
                successful_results.items(),
                key=lambda x: x[1].combined_score or 0,
            )
        elif metric == "fairness_improvement":
            best = max(
                successful_results.items(),
                key=lambda x: x[1].fairness_improvement or 0,
            )
        elif metric == "accuracy":
            best = max(
                successful_results.items(),
                key=lambda x: x[1].metrics_after.get("accuracy", 0),
            )
        else:
            best = list(successful_results.items())[0]

        return best[0], best[1]
