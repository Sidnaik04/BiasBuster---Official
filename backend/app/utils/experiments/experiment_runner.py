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

# Import mitigation strategies
from app.utils.mitigation.threshold import apply_threshold_optimizer
from app.utils.mitigation.reweighting import compute_sample_weights
from app.utils.mitigation.smote import apply_smote

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
        Run a single mitigation strategy.

        Args:
            strategy_name: 'threshold', 'reweighting', or 'smote'
            timeout_seconds: Maximum execution time

        Returns:
            Strategy result dictionary
        """
        result = {
            "strategy": strategy_name,
            "status": "success",
            "error": None,
        }

        start_time = time.time()

        try:
            logger.info(f"Starting strategy: {strategy_name}")

            # Prepare data for strategy
            sensitive_series = self.df[
                self.sensitive_columns[0]
            ]  # Use first sensitive column
            X_train = self.X
            y_train = self.y_true

            if isinstance(self.model, Pipeline):
                X_infer = X_train
            else:
                X_infer = encode_features_for_inference(X_train)

            # Run appropriate strategy
            if strategy_name == "threshold":
                mitigated_model = self._run_threshold_strategy(
                    X_train, y_train, X_infer, sensitive_series
                )

            elif strategy_name == "reweighting":
                mitigated_model = self._run_reweighting_strategy(
                    X_train, y_train, X_infer, sensitive_series
                )

            elif strategy_name == "smote":
                mitigated_model = self._run_smote_strategy(
                    X_train, y_train, X_infer, sensitive_series
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            # Get predictions with mitigated model
            y_pred_mitigated = predict_labels(mitigated_model, X_infer)
            y_pred_mitigated = np.nan_to_num(y_pred_mitigated).astype(int)

            # Compute metrics after mitigation
            accuracy_after = float(accuracy_score(self.y_true, y_pred_mitigated))

            # Fairness metrics
            fairness_metrics_after = {}
            all_dpds_after = []
            all_eods_after = []

            for sensitive_col in self.sensitive_columns:
                group_rates = {}
                group_tprs = {}

                for group in self.df[sensitive_col].unique():
                    mask = self.df[sensitive_col] == group
                    if mask.sum() > 0:
                        group_rates[group] = (y_pred_mitigated[mask] == 1).mean()
                        positives = self.y_true[mask] == 1
                        if positives.sum() > 0:
                            group_tprs[group] = (
                                y_pred_mitigated[mask][positives] == 1
                            ).mean()
                        else:
                            group_tprs[group] = 0.0

                dpd = demographic_parity_difference(group_rates)
                eod = equal_opportunity_difference(group_tprs) if group_tprs else 0.0

                fairness_metrics_after[sensitive_col] = {
                    "dpd": float(dpd),
                    "eod": float(eod),
                }

                all_dpds_after.append(dpd)
                all_eods_after.append(eod)

            avg_dpd_after = np.mean(all_dpds_after) if all_dpds_after else 0.0
            avg_eod_after = np.mean(all_eods_after) if all_eods_after else 0.0
            fairness_score_after = 1 - (abs(avg_dpd_after) + abs(avg_eod_after)) / 2
            fairness_score_after = max(0.0, fairness_score_after)

            # Compute improvements
            accuracy_drop = accuracy_after - self.metrics_before["accuracy"]
            fairness_improvement = (
                fairness_score_after - self.metrics_before["fairness_score"]
            )

            # Combined score: 0.6*fairness_improvement - 0.4*accuracy_drop
            combined_score = (0.6 * fairness_improvement) - (0.4 * abs(accuracy_drop))

            # Update result
            result.update(
                {
                    "accuracy_before": self.metrics_before["accuracy"],
                    "dpd_before": self.metrics_before["dpd"],
                    "eod_before": self.metrics_before["eod"],
                    "dir_before": self.metrics_before["dir"],
                    "fairness_score_before": self.metrics_before["fairness_score"],
                    "accuracy_after": accuracy_after,
                    "dpd_after": avg_dpd_after,
                    "eod_after": avg_eod_after,
                    "dir_after": (
                        float(
                            np.mean(
                                [
                                    fairness_metrics_after[col]["dpd"]
                                    for col in fairness_metrics_after
                                ]
                            )
                        )
                        if fairness_metrics_after
                        else 1.0
                    ),
                    "fairness_score_after": fairness_score_after,
                    "accuracy_drop": accuracy_drop,
                    "fairness_improvement": fairness_improvement,
                    "combined_score": combined_score,
                }
            )

            duration = time.time() - start_time
            result["duration_seconds"] = duration

            logger.info(
                f"Strategy {strategy_name} completed: fairness_improvement={fairness_improvement:.4f}, accuracy_drop={accuracy_drop:.4f}"
            )

        except TimeoutException as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.warning(f"Strategy {strategy_name} timed out: {e}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Strategy {strategy_name} failed: {e}")

        return result

    def _run_threshold_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """Apply threshold optimization strategy."""
        try:
            return apply_threshold_optimizer(
                self.model,
                X_train,
                y_train,
                sensitive_series,
                grid_size=200,
            )
        except Exception as e:
            logger.error(f"Threshold strategy failed: {e}")
            raise

    def _run_reweighting_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """Apply reweighting strategy."""
        try:
            from sklearn.utils.class_weight import (
                compute_sample_weight as sklearn_compute_sample_weight,
            )

            weights = compute_sample_weights(sensitive_series)

            # Clone model and fit with weights
            if isinstance(self.model, Pipeline):
                model_copy = self.model
                if hasattr(model_copy, "fit"):
                    model_copy.fit(
                        X_train,
                        y_train,
                        **{f"{model_copy.steps[-1][0]}__sample_weight": weights},
                    )
            else:
                from sklearn.base import clone

                model_copy = clone(self.model)
                model_copy.fit(X_train, y_train, sample_weight=weights)

            return model_copy
        except Exception as e:
            logger.error(f"Reweighting strategy failed: {e}")
            raise

    def _run_smote_strategy(self, X_train, y_train, X_infer, sensitive_series):
        """Apply SMOTE strategy."""
        try:
            X_resampled, y_resampled = apply_smote(X_train, y_train)

            # Clone model and fit with resampled data
            if isinstance(self.model, Pipeline):
                model_copy = self.model
                model_copy.fit(X_resampled, y_resampled)
            else:
                from sklearn.base import clone

                model_copy = clone(self.model)
                model_copy.fit(X_resampled, y_resampled)

            return model_copy
        except Exception as e:
            logger.error(f"SMOTE strategy failed: {e}")
            raise

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
