"""Reweighting strategy implementation."""

import logging
import numpy as np
import pandas as pd
from sklearn.base import clone

from .base_strategy import BaseMitigationStrategy

logger = logging.getLogger(__name__)


def compute_sample_weights(y_train, sensitive_features_train, target_column="target"):
    """
    Compute sample weights to balance classes and sensitive groups.

    Args:
        y_train: Training labels
        sensitive_features_train: Sensitive features
        target_column: Target column name (unused, for compatibility)

    Returns:
        Sample weights array
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Handle different input types
    if isinstance(y_train, pd.Series):
        y_array = y_train.values
    else:
        y_array = np.asarray(y_train)

    if isinstance(sensitive_features_train, pd.DataFrame):
        sensitive_array = sensitive_features_train.iloc[:, 0].values
    elif isinstance(sensitive_features_train, pd.Series):
        sensitive_array = sensitive_features_train.values
    else:
        sensitive_array = np.asarray(sensitive_features_train).flatten()

    # Compute class weights
    classes = np.unique(y_array)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_array)
    class_weight_dict = dict(zip(classes, class_weights))

    # Get base weights by class
    base_weights = np.array([class_weight_dict[y] for y in y_array])

    # Compute group weights (balance sensitive groups)
    unique_groups = np.unique(sensitive_array)
    group_weights = {}

    for group in unique_groups:
        mask = sensitive_array == group
        group_weight = 1.0 / np.sum(mask)
        group_weights[group] = group_weight

    group_based_weights = np.array([group_weights[s] for s in sensitive_array])
    group_based_weights = group_based_weights / np.mean(group_based_weights)

    # Combine class and group weights
    combined_weights = base_weights * group_based_weights
    combined_weights = combined_weights / np.mean(combined_weights)

    return combined_weights


class ReweightingStrategy(BaseMitigationStrategy):
    """Reweighting strategy using sample weights."""

    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
        target_column,
    ):
        """
        Initialize reweighting strategy.

        Args:
            model: Base sklearn model with sample_weight support
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
        """
        super().__init__(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sensitive_features_train=sensitive_features_train,
            sensitive_features_test=sensitive_features_test,
            target_column=target_column,
            strategy_name="reweighting",
        )
        self.sample_weights = None

    def fit(self) -> bool:
        """
        Fit model with reweighted samples.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs(
                self.model,
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.sensitive_features_train,
                self.sensitive_features_test,
            )

            if not is_valid:
                logger.error(f"Reweighting strategy validation failed: {error_msg}")
                self.diagnostics["error"] = error_msg
                return False

            # Compute sample weights
            self.sample_weights = compute_sample_weights(
                self.y_train, self.sensitive_features_train, self.target_column
            )

            self.diagnostics["sample_weights_stats"] = {
                "mean": float(np.mean(self.sample_weights)),
                "std": float(np.std(self.sample_weights)),
                "min": float(np.min(self.sample_weights)),
                "max": float(np.max(self.sample_weights)),
            }

            logger.info(
                f"Reweighting: Computed weights - mean={np.mean(self.sample_weights):.4f}, "
                f"std={np.std(self.sample_weights):.4f}"
            )

            # Clone and fit model with sample weights
            mitigated_model = clone(self.model)

            # Check if model supports sample_weight
            if not hasattr(mitigated_model, "fit"):
                logger.error("Model does not have fit method")
                self.diagnostics["error"] = "Model does not have fit method"
                return False

            # Fit with sample weights
            try:
                mitigated_model.fit(
                    self.X_train, self.y_train, sample_weight=self.sample_weights
                )
            except TypeError:
                # Model doesn't support sample_weight, fit without it
                logger.warning(
                    "Model does not support sample_weight parameter, fitting without weights"
                )
                mitigated_model.fit(self.X_train, self.y_train)

            self.mitigated_model = mitigated_model
            self.diagnostics["fit_success"] = True
            logger.info("Reweighting strategy fitted successfully")
            return True

        except Exception as e:
            logger.error(f"Reweighting strategy fit failed: {str(e)}")
            self.diagnostics["error"] = str(e)
            return False

    def predict(self, X: np.ndarray, sensitive_features=None) -> np.ndarray:
        """
        Make predictions using reweighted model.

        Args:
            X: Feature matrix
            sensitive_features: Sensitive features (not used for reweighting)

        Returns:
            Predictions
        """
        if self.mitigated_model is None:
            raise RuntimeError("Reweighting model not fitted yet")

        return self.mitigated_model.predict(X)

    def _requires_sensitive_features_for_prediction(self) -> bool:
        """Reweighting doesn't require sensitive features for prediction."""
        return False
