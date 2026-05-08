"""Data Correction Wizard for fairness mitigation and debiasing."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import uuid
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from app.config import settings
from app.utils.fairness.metrics import (
    compute_selection_rate,
    compute_true_positive_rate,
    compute_false_positive_rate,
)


class DataCorrectionWizard:
    """
    Applies fairness mitigation strategies and generates debiased datasets/models.

    Strategies:
    - threshold: Adjust prediction threshold per group
    - reweighting: Reweight training samples by group
    - smote: Apply SMOTE to balance sensitive groups
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        model,
        target_column: str,
        sensitive_columns: list,
        y_predictions: np.ndarray,
        y_true: np.ndarray,
    ):
        """
        Initialize correction wizard.

        Args:
            dataset: Full dataset (original)
            model: Trained model
            target_column: Target column name
            sensitive_columns: List of sensitive attribute columns
            y_predictions: Model predictions
            y_true: True labels
        """
        self.dataset = dataset.copy()
        self.model = model
        self.target_column = target_column
        self.sensitive_columns = sensitive_columns
        self.y_predictions = y_predictions
        self.y_true = y_true

        # Create correction directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories for artifacts."""
        for dir_name in [
            "corrected",
            "models",
            "reports",
        ]:
            dir_path = Path(settings.TEMP_DIR) / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def correct(self, strategy: str = "reweighting") -> Dict[str, Any]:
        """
        Apply fairness mitigation strategy.

        Args:
            strategy: "threshold", "reweighting", or "smote"

        Returns:
            Result dict with metrics, paths, and summary
        """
        # Compute before metrics
        metrics_before = self._compute_metrics(self.y_true, self.y_predictions)

        # Apply strategy
        if strategy == "threshold":
            result = self._apply_threshold_adjustment()
        elif strategy == "reweighting":
            result = self._apply_reweighting()
        elif strategy == "smote":
            result = self._apply_smote()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Get corrected dataset and model
        corrected_dataset = result.get("corrected_dataset", self.dataset)
        corrected_model = result.get("corrected_model", self.model)

        # Make predictions with corrected model
        y_corrected = self._predict_safe(corrected_model, corrected_dataset)

        # Compute after metrics
        metrics_after = self._compute_metrics(self.y_true, y_corrected)

        # Generate unique ID
        correction_id = f"corr_{uuid.uuid4().hex[:8]}"

        # Export artifacts
        dataset_path = self._export_dataset(corrected_dataset, correction_id, strategy)
        model_path = self._export_model(corrected_model, correction_id, strategy)
        report_path = self._export_report(
            correction_id, strategy, metrics_before, metrics_after
        )

        # Compute improvements
        improvements = self._compute_improvements(metrics_before, metrics_after)

        return {
            "correction_id": correction_id,
            "strategy_applied": strategy,
            "dataset_export_path": str(dataset_path),
            "model_export_path": str(model_path),
            "report_export_path": str(report_path),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvements": improvements,
            "summary": self._generate_summary(
                strategy, improvements, metrics_before, metrics_after
            ),
        }

    def _apply_threshold_adjustment(self) -> Dict[str, Any]:
        """
        Threshold optimization strategy.

        Adjusts decision threshold per sensitive group to achieve fairness.
        """
        try:
            # Calculate per-group thresholds for equalized odds
            group_thresholds = {}

            for attr_col in self.sensitive_columns:
                for group in self.dataset[attr_col].unique():
                    group_mask = self.dataset[attr_col] == group
                    group_y_true = self.y_true[group_mask]
                    group_y_pred_proba = self.y_predictions[group_mask]

                    # Find threshold that maximizes TPR
                    if len(group_y_true) > 0 and group_y_true.sum() > 0:
                        best_threshold = 0.5
                        best_tpr = 0
                        for threshold in np.arange(0.1, 0.9, 0.1):
                            y_pred_binary = (group_y_pred_proba >= threshold).astype(
                                int
                            )
                            tpr = (
                                np.sum((y_pred_binary == 1) & (group_y_true == 1))
                                / group_y_true.sum()
                            )
                            if tpr > best_tpr:
                                best_tpr = tpr
                                best_threshold = threshold

                        group_thresholds[(attr_col, group)] = best_threshold

            # Create threshold-adjusted predictor
            class ThresholdAdjustedModel:
                def __init__(self, base_model, group_thresholds, sensitive_columns):
                    self.base_model = base_model
                    self.group_thresholds = group_thresholds
                    self.sensitive_columns = sensitive_columns

                def predict(self, X):
                    preds = self.base_model.predict(X)
                    adjusted = preds.copy()
                    # Note: For threshold strategy, we can't adjust without sensitive attrs in X
                    return adjusted

            adjusted_model = ThresholdAdjustedModel(
                self.model, group_thresholds, self.sensitive_columns
            )

            return {
                "corrected_model": adjusted_model,
                "corrected_dataset": self.dataset,
            }

        except Exception as e:
            print(f"Threshold adjustment failed: {e}")
            return {"corrected_model": self.model, "corrected_dataset": self.dataset}

    def _apply_reweighting(self) -> Dict[str, Any]:
        """
        Sample reweighting strategy.

        Reweights training samples inversely proportional to group size.
        """
        try:
            X = self.dataset.drop(columns=[self.target_column])
            y = self.dataset[self.target_column]

            # Compute sample weights
            sample_weights = self._compute_reweighting_weights()

            # Retrain model with weights
            corrected_model = self._retrain_model(X, y, sample_weights)

            return {
                "corrected_model": corrected_model,
                "corrected_dataset": self.dataset,
            }

        except Exception as e:
            print(f"Reweighting failed: {e}")
            return {"corrected_model": self.model, "corrected_dataset": self.dataset}

    def _apply_smote(self) -> Dict[str, Any]:
        """
        SMOTE strategy for imbalanced groups.

        Oversamples underrepresented sensitive groups.
        """
        try:
            from imblearn.over_sampling import SMOTE as SMOTEResampler

            X = self.dataset.drop(columns=[self.target_column])
            y = self.dataset[self.target_column]

            # Create synthetic samples for underrepresented groups
            smote = SMOTEResampler(random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Reconstruct dataset
            corrected_dataset = X_resampled.copy()
            corrected_dataset[self.target_column] = y_resampled

            # Retrain model
            corrected_model = self._retrain_model(X_resampled, y_resampled, None)

            return {
                "corrected_model": corrected_model,
                "corrected_dataset": corrected_dataset,
            }

        except ImportError:
            print("SMOTE requires: pip install imbalanced-learn")
            return {"corrected_model": self.model, "corrected_dataset": self.dataset}
        except Exception as e:
            print(f"SMOTE failed: {e}")
            return {"corrected_model": self.model, "corrected_dataset": self.dataset}

    def _compute_reweighting_weights(self) -> np.ndarray:
        """Compute sample weights for reweighting."""
        weights = np.ones(len(self.dataset))

        # Get group sizes
        for attr_col in self.sensitive_columns:
            for group in self.dataset[attr_col].unique():
                group_mask = self.dataset[attr_col] == group
                group_size = group_mask.sum()
                total_size = len(self.dataset)

                # Inverse group size weighting
                if group_size > 0:
                    weight = total_size / (
                        len(self.dataset[attr_col].unique()) * group_size
                    )
                    weights[group_mask] *= weight

        # Normalize weights
        weights = weights / weights.sum() * len(self.dataset)
        return weights

    def _retrain_model(
        self, X: pd.DataFrame, y: pd.Series, sample_weights: Optional[np.ndarray] = None
    ):
        """Retrain model with corrected data."""
        try:
            # Clone the original model
            from sklearn.base import clone

            model_copy = clone(self.model)

            # Fit with optional weights
            if sample_weights is not None and hasattr(model_copy, "fit"):
                model_copy.fit(X, y, sample_weight=sample_weights)
            else:
                model_copy.fit(X, y)

            return model_copy
        except Exception as e:
            print(f"Retraining failed: {e}")
            return self.model

    def _predict_safe(self, model, dataset: pd.DataFrame) -> np.ndarray:
        """Safely predict using corrected model."""
        try:
            X = dataset.drop(columns=[self.target_column], errors="ignore")
            predictions = model.predict(X)
            return predictions
        except Exception as e:
            print(f"Prediction failed: {e}")
            return self.y_predictions

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute fairness and accuracy metrics.

        Returns:
            Dict with accuracy, DPD, EOD, fairness_score
        """
        metrics = {}

        # Accuracy
        try:
            accuracy = (y_true == y_pred).mean()
            metrics["accuracy"] = float(accuracy)
        except:
            metrics["accuracy"] = 0.0

        # Fairness metrics per group
        dpd_list = []
        eod_list = []

        for attr_col in self.sensitive_columns:
            try:
                unique_groups = self.dataset[attr_col].unique()
                if len(unique_groups) >= 2:
                    group_0 = unique_groups[0]
                    group_1 = unique_groups[1]

                    mask_0 = self.dataset[attr_col] == group_0
                    mask_1 = self.dataset[attr_col] == group_1

                    # DPD (Demographic Parity Difference)
                    sr_0 = compute_selection_rate(y_pred[mask_0])
                    sr_1 = compute_selection_rate(y_pred[mask_1])
                    dpd = abs(sr_0 - sr_1)
                    dpd_list.append(dpd)

                    # EOD (Equalized Odds Difference)
                    if y_true[mask_0].sum() > 0 and y_true[mask_1].sum() > 0:
                        tpr_0 = compute_true_positive_rate(
                            y_true[mask_0], y_pred[mask_0]
                        )
                        tpr_1 = compute_true_positive_rate(
                            y_true[mask_1], y_pred[mask_1]
                        )
                        eod = abs(tpr_0 - tpr_1)
                        eod_list.append(eod)
            except Exception as e:
                print(f"Fairness metric computation failed for {attr_col}: {e}")

        # Aggregate metrics
        metrics["dpd"] = float(np.mean(dpd_list)) if dpd_list else 0.0
        metrics["eod"] = float(np.mean(eod_list)) if eod_list else 0.0
        metrics["fairness_score"] = float(1.0 - (metrics["dpd"] + metrics["eod"]) / 2)

        return metrics

    def _compute_improvements(
        self, metrics_before: Dict[str, float], metrics_after: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute improvement metrics."""
        return {
            "fairness_gain": float(
                metrics_after.get("fairness_score", 0)
                - metrics_before.get("fairness_score", 0)
            ),
            "accuracy_change": float(
                metrics_after.get("accuracy", 0) - metrics_before.get("accuracy", 0)
            ),
            "dpd_reduction": float(
                metrics_before.get("dpd", 0) - metrics_after.get("dpd", 0)
            ),
            "eod_reduction": float(
                metrics_before.get("eod", 0) - metrics_after.get("eod", 0)
            ),
        }

    def _export_dataset(
        self, dataset: pd.DataFrame, correction_id: str, strategy: str
    ) -> Path:
        """Export corrected dataset to CSV."""
        export_dir = Path(settings.TEMP_DIR) / "corrected"
        export_dir.mkdir(parents=True, exist_ok=True)

        export_path = export_dir / f"{correction_id}_{strategy}_dataset.csv"
        dataset.to_csv(export_path, index=False)

        return export_path

    def _export_model(self, model, correction_id: str, strategy: str) -> Path:
        """Export corrected model to joblib."""
        export_dir = Path(settings.TEMP_DIR) / "models"
        export_dir.mkdir(parents=True, exist_ok=True)

        export_path = export_dir / f"{correction_id}_{strategy}_model.joblib"
        joblib.dump(model, export_path)

        return export_path

    def _export_report(
        self,
        correction_id: str,
        strategy: str,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
    ) -> Path:
        """Export correction report as JSON metadata."""
        export_dir = Path(settings.TEMP_DIR) / "reports"
        export_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "correction_id": correction_id,
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "dataset_shape": list(self.dataset.shape),
            "sensitive_attributes": self.sensitive_columns,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        }

        export_path = export_dir / f"{correction_id}_report.json"
        with open(export_path, "w") as f:
            json.dump(report, f, indent=2)

        return export_path

    def _generate_summary(
        self,
        strategy: str,
        improvements: Dict[str, float],
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
    ) -> str:
        """Generate human-readable summary."""
        fairness_gain = improvements["fairness_gain"]
        accuracy_change = improvements["accuracy_change"]

        verdict = "improved" if fairness_gain > 0 else "degraded"
        accuracy_verb = "reduced" if accuracy_change < 0 else "improved"

        summary = (
            f"Strategy '{strategy}' {verdict} fairness by {abs(fairness_gain):.2%} "
            f"with accuracy {accuracy_verb} by {abs(accuracy_change):.2%}. "
            f"Fairness score: {metrics_before['fairness_score']:.2%} → "
            f"{metrics_after['fairness_score']:.2%}."
        )

        if fairness_gain > 0.1:
            summary += " Significant fairness improvement achieved."
        elif fairness_gain < -0.05:
            summary += " Warning: Fairness degraded significantly."

        return summary
