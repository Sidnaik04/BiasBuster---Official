"""
Auto Experimentation Engine: Automatically runs and compares mitigation strategies.

Pipeline:
1. Load dataset & model
2. Compute baseline metrics (accuracy + fairness)
3. Run each strategy sequentially with timeout
4. Evaluate results
5. Select best strategy
6. Generate insights
7. Store results
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
import signal
from contextlib import contextmanager

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from app.utils.dataset_loader import load_dataset
from app.utils.model_loader import load_model
from app.utils.prediction import predict_labels
from app.utils.dataset_validation import validate_dataset_health
from app.utils.target_encoder import encode_target_column
from app.utils.feature_encoder import encode_features_for_inference
from app.utils.sensitive_validation import validate_sensitive_columns
from app.utils.sensitive_preprocessing import bin_age_column
from app.utils.fairness_metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    disparate_impact_ratio,
    true_positive_rate,
)

# Import mitigation orchestrator
from app.services.mitigation_orchestrator import MitigationOrchestrator

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when strategy execution times out."""

    pass


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for handling timeout."""

    def handle_timeout(signum, frame):
        raise TimeoutException(f"Strategy execution timed out after {seconds} seconds")

    # Only works on Unix systems
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(seconds)
        yield
        signal.alarm(0)  # Disable alarm
    except (ValueError, OSError):
        # Fallback: no timeout on Windows
        yield


class ExperimentRunner:
    """Orchestrates experiment pipeline."""

    def __init__(
        self,
        dataset_filename: str,
        model_filename: str,
        target_column: str,
        sensitive_columns: List[str],
    ):
        """
        Initialize experiment runner.

        Args:
            dataset_filename: Path to dataset file
            model_filename: Path to model file
            target_column: Target column name
            sensitive_columns: List of sensitive attribute names
        """
        self.dataset_filename = dataset_filename
        self.model_filename = model_filename
        self.target_column = target_column
        self.sensitive_columns = sensitive_columns

        self.df = None
        self.model = None
        self.X = None
        self.y_true = None
        self.y_pred_baseline = None
        self.metrics_before = None
        self.warnings = []

        # Initialize orchestrator
        self.orchestrator = MitigationOrchestrator()

    def load_and_prepare_data(self) -> bool:
        """
        Load dataset and model, perform preprocessing.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_filename}")
            self.df = load_dataset(self.dataset_filename)

            logger.info(f"Loading model: {self.model_filename}")
            self.model = load_model(self.model_filename)

            # Dataset health check
            dataset_health = validate_dataset_health(self.df)

            # Encode target
            self.df, target_info = encode_target_column(self.df, self.target_column)
            self.y_true = self.df[self.target_column].astype(int)

            # Validate sensitive columns
            sensitive_info = validate_sensitive_columns(self.df, self.sensitive_columns)

            # Bin age if present
            for col in self.sensitive_columns:
                if col.lower() == "age":
                    self.df = bin_age_column(self.df, col)
                    idx = self.sensitive_columns.index(col)
                    self.sensitive_columns[idx] = col + "_group"
                    break

            # Convert sensitive columns to string
            for col in self.sensitive_columns:
                self.df[col] = self.df[col].astype(str)

            # Prepare features
            self.X = self.df.drop(columns=[self.target_column])

            if isinstance(self.model, Pipeline):
                X_infer = self.X
            else:
                X_infer = encode_features_for_inference(self.X)

            # Get baseline predictions
            self.y_pred_baseline = predict_labels(self.model, X_infer)
            self.y_pred_baseline = np.nan_to_num(self.y_pred_baseline).astype(int)

            logger.info("Data loading and preparation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.warnings.append(f"Data loading error: {str(e)}")
            return False

    def compute_baseline_metrics(self) -> Dict[str, Any]:
        """
        Compute baseline metrics before any mitigation.

        Returns:
            Dictionary with accuracy, fairness metrics
        """
        try:
            # Accuracy
            accuracy = float(accuracy_score(self.y_true, self.y_pred_baseline))

            # Fairness metrics per sensitive attribute
            fairness_metrics = {}
            all_dpds = []
            all_eods = []

            for sensitive_col in self.sensitive_columns:
                group_rates = {}
                group_tprs = {}

                for group in self.df[sensitive_col].unique():
                    mask = self.df[sensitive_col] == group
                    if mask.sum() > 0:
                        group_rates[group] = (self.y_pred_baseline[mask] == 1).mean()
                        positives = self.y_true[mask] == 1
                        if positives.sum() > 0:
                            group_tprs[group] = (
                                self.y_pred_baseline[mask][positives] == 1
                            ).mean()
                        else:
                            group_tprs[group] = 0.0

                dpd = demographic_parity_difference(group_rates)
                eod = equal_opportunity_difference(group_tprs) if group_tprs else 0.0
                dir_val = disparate_impact_ratio(group_rates) if group_rates else 1.0

                fairness_metrics[sensitive_col] = {
                    "dpd": float(dpd),
                    "eod": float(eod),
                    "dir": float(dir_val),
                }

                all_dpds.append(dpd)
                all_eods.append(eod)

            # Aggregate fairness score: 1 - (avg(|DPD| + |EOD|)/2)
            avg_dpd = np.mean(all_dpds) if all_dpds else 0.0
            avg_eod = np.mean(all_eods) if all_eods else 0.0
            fairness_score = 1 - (abs(avg_dpd) + abs(avg_eod)) / 2
            fairness_score = max(0.0, fairness_score)  # Ensure >= 0

            self.metrics_before = {
                "accuracy": accuracy,
                "dpd": float(np.mean(all_dpds)),
                "eod": float(np.mean(all_eods)),
                "dir": float(
                    np.mean([fairness_metrics[col]["dir"] for col in fairness_metrics])
                ),
                "fairness_score": fairness_score,
                "by_attribute": fairness_metrics,
            }

            logger.info(
                f"Baseline metrics computed: accuracy={accuracy:.4f}, fairness_score={fairness_score:.4f}"
            )
            return self.metrics_before

        except Exception as e:
            logger.error(f"Failed to compute baseline metrics: {e}")
            raise

    def run_strategy(
        self, strategy_name: str, timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Run a single mitigation strategy using orchestrator.

        Args:
            strategy_name: 'threshold', 'reweighting', or 'smote'
            timeout_seconds: Maximum execution time (note: not currently enforced by orchestrator)

        Returns:
            Strategy result dictionary compatible with legacy experiment format
        """
        try:
            logger.info(f"Starting strategy via orchestrator: {strategy_name}")

            # Prepare test data (same as training data in this context)
            sensitive_test = self.df[
                self.sensitive_columns[0]
            ]  # Use first sensitive column

            # Prepare features for inference
            if isinstance(self.model, Pipeline):
                X_infer = self.X
            else:
                X_infer = encode_features_for_inference(self.X)

            # Define metrics computation function
            def compute_metrics(model, X_test, y_test, sensitive_features):
                """Compute fairness and accuracy metrics."""
                try:
                    y_pred = predict_labels(model, X_test)
                    y_pred = np.nan_to_num(y_pred).astype(int)

                    accuracy = float(accuracy_score(y_test, y_pred))

                    # Compute fairness metrics per sensitive column
                    group_rates = {}
                    group_tprs = {}

                    for group in self.df[self.sensitive_columns[0]].unique():
                        mask = self.df[self.sensitive_columns[0]] == group
                        if mask.sum() > 0:
                            group_rates[group] = (y_pred[mask] == 1).mean()
                            positives = y_test[mask] == 1
                            if positives.sum() > 0:
                                group_tprs[group] = (
                                    y_pred[mask][positives] == 1
                                ).mean()
                            else:
                                group_tprs[group] = 0.0

                    dpd = demographic_parity_difference(group_rates)
                    eod = (
                        equal_opportunity_difference(group_tprs) if group_tprs else 0.0
                    )
                    dir_val = disparate_impact_ratio(group_rates)

                    return {
                        "accuracy": accuracy,
                        "dpd": float(dpd),
                        "eod": float(eod),
                        "dir": float(dir_val),
                    }
                except Exception as e:
                    logger.error(f"Failed to compute metrics: {e}")
                    return {
                        "accuracy": 0.0,
                        "dpd": 0.0,
                        "eod": 0.0,
                        "dir": 1.0,
                    }

            # Use orchestrator to run strategy
            mitigation_result = self.orchestrator.run_strategy(
                strategy_name=strategy_name,
                model=self.model,
                X_train=self.X,
                y_train=self.y_true,
                X_test=self.X,
                y_test=self.y_true,
                sensitive_features_train=sensitive_test,
                sensitive_features_test=sensitive_test,
                target_column=self.target_column,
                compute_metrics_func=compute_metrics,
            )

            # Convert orchestrator result to legacy format
            result = {
                "strategy": strategy_name,
                "status": (
                    "success" if mitigation_result.status == "success" else "failed"
                ),
                "error": mitigation_result.error_message,
                "accuracy_before": self.metrics_before["accuracy"],
                "dpd_before": self.metrics_before["dpd"],
                "eod_before": self.metrics_before["eod"],
                "dir_before": self.metrics_before["dir"],
                "fairness_score_before": self.metrics_before["fairness_score"],
                "accuracy_after": mitigation_result.metrics_after.get("accuracy", 0.0),
                "dpd_after": mitigation_result.metrics_after.get("dpd", 0.0),
                "eod_after": mitigation_result.metrics_after.get("eod", 0.0),
                "dir_after": mitigation_result.metrics_after.get("dir", 1.0),
                "fairness_score_after": 1
                - (
                    abs(mitigation_result.metrics_after.get("dpd", 0.0))
                    + abs(mitigation_result.metrics_after.get("eod", 0.0))
                )
                / 2,
                "accuracy_drop": (
                    mitigation_result.metrics_after.get("accuracy", 0.0)
                    - self.metrics_before["accuracy"]
                ),
                "fairness_improvement": mitigation_result.fairness_improvement or 0.0,
                "combined_score": mitigation_result.combined_score or 0.0,
                "duration_seconds": mitigation_result.execution_time_seconds or 0.0,
            }

            logger.info(
                f"Strategy {strategy_name} completed: fairness_improvement={result['fairness_improvement']:.4f}, "
                f"accuracy_drop={result['accuracy_drop']:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Strategy {strategy_name} execution failed: {e}")
            return {
                "strategy": strategy_name,
                "status": "failed",
                "error": str(e),
                "accuracy_before": (
                    self.metrics_before["accuracy"] if self.metrics_before else 0.0
                ),
                "dpd_before": (
                    self.metrics_before["dpd"] if self.metrics_before else 0.0
                ),
                "eod_before": (
                    self.metrics_before["eod"] if self.metrics_before else 0.0
                ),
                "dir_before": (
                    self.metrics_before["dir"] if self.metrics_before else 1.0
                ),
                "fairness_score_before": (
                    self.metrics_before["fairness_score"]
                    if self.metrics_before
                    else 0.0
                ),
                "duration_seconds": 0.0,
            }

    def _run_threshold_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """DEPRECATED: Use orchestrator instead."""
        raise NotImplementedError("Use orchestrator.run_strategy() instead")

    def _run_reweighting_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """DEPRECATED: Use orchestrator instead."""
        raise NotImplementedError("Use orchestrator.run_strategy() instead")

    def _run_smote_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """DEPRECATED: Use orchestrator instead."""
        raise NotImplementedError("Use orchestrator.run_strategy() instead")

    @staticmethod
    def select_best_strategy(
        results: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Select best strategy based on combined score.

        combined_score = (0.6 * fairness_improvement) - (0.4 * |accuracy_drop|)

        Args:
            results: List of strategy results

        Returns:
            (best_strategy_name, combined_score) or (None, None) if all failed
        """
        successful_results = [r for r in results if r["status"] == "success"]

        if not successful_results:
            logger.warning("No successful strategies to select from")
            return None, None

        # Sort by combined score descending
        sorted_results = sorted(
            successful_results,
            key=lambda x: x.get("combined_score", -float("inf")),
            reverse=True,
        )

        best = sorted_results[0]
        best_strategy = best["strategy"]
        best_score = best["combined_score"]

        logger.info(f"Best strategy: {best_strategy} with score {best_score:.4f}")

        return best_strategy, best_score

    @staticmethod
    def generate_insights(
        results: List[Dict[str, Any]], best_strategy: Optional[str]
    ) -> str:
        """
        Generate human-readable insights from experiment results.

        Args:
            results: List of strategy results
            best_strategy: Name of best strategy

        Returns:
            Insight string
        """
        if not best_strategy:
            return "All strategies failed. Please check logs for details."

        best_result = next((r for r in results if r["strategy"] == best_strategy), None)
        if not best_result:
            return "Unable to generate insights."

        insights = []

        # Fairness improvement
        fairness_imp = best_result.get("fairness_improvement", 0)
        if fairness_imp > 0.1:
            insights.append(
                f"{best_strategy.capitalize()} achieved significant fairness improvement (+{fairness_imp:.1%})."
            )
        elif fairness_imp > 0:
            insights.append(
                f"{best_strategy.capitalize()} achieved modest fairness improvement (+{fairness_imp:.1%})."
            )
        else:
            insights.append(f"{best_strategy.capitalize()} did not improve fairness.")

        # Accuracy impact
        accuracy_drop = best_result.get("accuracy_drop", 0)
        if accuracy_drop < -0.01:
            insights.append(f"Accuracy improved by {abs(accuracy_drop):.1%}.")
        elif accuracy_drop > 0.05:
            insights.append(
                f"Accuracy dropped by {accuracy_drop:.1%}, which may be acceptable given fairness gains."
            )
        elif accuracy_drop > 0:
            insights.append(f"Accuracy trade-off was minimal ({accuracy_drop:.1%}).")
        else:
            insights.append("Accuracy was maintained or improved.")

        # Strategy explanation
        if best_strategy == "threshold":
            insights.append(
                "Threshold adjustment modified decision thresholds per demographic group to equalize opportunity."
            )
        elif best_strategy == "reweighting":
            insights.append(
                "Sample reweighting balanced underrepresented groups in training."
            )
        elif best_strategy == "smote":
            insights.append(
                "SMOTE oversampled minority class to address class imbalance."
            )

        # Overall recommendation
        if fairness_imp > 0.1 and abs(accuracy_drop) < 0.05:
            insights.append("This strategy is recommended for deployment.")
        elif fairness_imp > 0 and abs(accuracy_drop) < 0.1:
            insights.append(
                "This strategy offers a reasonable fairness-accuracy trade-off."
            )
        else:
            insights.append(
                "Consider testing additional strategies or tuning parameters."
            )

        return " ".join(insights)

    def run_experiment(
        self, strategies: List[str] = None, timeout_per_strategy: int = 300
    ) -> Dict[str, Any]:
        """
        Execute full experiment pipeline.

        Args:
            strategies: List of strategies to test (default: all)
            timeout_per_strategy: Timeout in seconds per strategy

        Returns:
            Complete experiment report
        """
        if strategies is None:
            strategies = ["threshold", "reweighting", "smote"]

        experiment_id = str(uuid4())
        start_time = time.time()

        try:
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data():
                return {
                    "experiment_id": experiment_id,
                    "strategies_tested": [],
                    "results": [],
                    "best_strategy": None,
                    "insights": "Experiment failed during data loading.",
                    "warnings": self.warnings,
                    "status": "failed",
                    "error_message": "Data loading failed",
                }

            # Step 2: Compute baseline metrics
            self.compute_baseline_metrics()

            # Step 3: Run each strategy
            results = []
            for strategy in strategies:
                logger.info(f"Running strategy: {strategy}")
                result = self.run_strategy(
                    strategy, timeout_seconds=timeout_per_strategy
                )
                results.append(result)

            # Count successful strategies
            successful_count = sum(1 for r in results if r["status"] == "success")
            logger.info(
                f"Completed: {successful_count}/{len(strategies)} strategies successful"
            )

            # Step 4: Select best strategy
            best_strategy, best_score = self.select_best_strategy(results)

            # Step 5: Generate insights
            insights = self.generate_insights(results, best_strategy)

            # Step 6: Build experiment report
            total_duration = time.time() - start_time

            experiment_report = {
                "experiment_id": experiment_id,
                "strategies_tested": strategies,
                "results": results,
                "metrics_before": self.metrics_before,
                "best_strategy": best_strategy,
                "best_strategy_score": best_score,
                "insights": insights,
                "warnings": self.warnings,
                "total_duration_seconds": total_duration,
                "status": "completed" if successful_count > 0 else "failed",
            }

            logger.info(
                f"Experiment {experiment_id} completed in {total_duration:.2f}s"
            )

            return experiment_report

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            total_duration = time.time() - start_time

            return {
                "experiment_id": experiment_id,
                "strategies_tested": strategies,
                "results": [],
                "best_strategy": None,
                "insights": f"Experiment failed: {str(e)}",
                "warnings": self.warnings,
                "total_duration_seconds": total_duration,
                "status": "failed",
                "error_message": str(e),
            }
