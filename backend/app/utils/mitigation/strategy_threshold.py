"""Threshold strategy implementation."""

import logging
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import clone

from .base_strategy import BaseMitigationStrategy

logger = logging.getLogger(__name__)


class ThresholdStrategy(BaseMitigationStrategy):
    """Fairlearn ThresholdOptimizer strategy."""

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
        grid_size: int = 200,
    ):
        """
        Initialize threshold strategy.

        Args:
            model: Base sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
            grid_size: Grid size for threshold search
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
            strategy_name="threshold",
        )
        self.grid_size = grid_size

    def fit(self) -> bool:
        """
        Fit threshold optimizer.

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
                logger.error(f"Threshold strategy validation failed: {error_msg}")
                self.diagnostics["error"] = error_msg
                return False

            # Extract base model if already wrapped
            if isinstance(self.model, ThresholdOptimizer):
                base_model = self.model.estimator_
            else:
                base_model = clone(self.model)

            # Determine predict method
            predict_method = (
                "predict_proba"
                if hasattr(base_model, "predict_proba")
                else "decision_function"
            )

            logger.info(
                f"Threshold: Using predict_method={predict_method}, grid_size={self.grid_size}"
            )

            # Create and fit ThresholdOptimizer
            optimizer = ThresholdOptimizer(
                estimator=base_model,
                constraints="equalized_odds",
                predict_method=predict_method,
                grid_size=self.grid_size,
            )

            # Fit on training data with sensitive features
            optimizer.fit(
                self.X_train,
                self.y_train,
                sensitive_features=self.sensitive_features_train,
            )

            self.mitigated_model = optimizer
            self.diagnostics["fit_success"] = True
            logger.info("Threshold strategy fitted successfully")
            return True

        except Exception as e:
            logger.error(f"Threshold strategy fit failed: {str(e)}")
            self.diagnostics["error"] = str(e)
            return False

    def predict(self, X: np.ndarray, sensitive_features=None) -> np.ndarray:
        """
        Make predictions using ThresholdOptimizer.

        Args:
            X: Feature matrix
            sensitive_features: Sensitive features (REQUIRED for ThresholdOptimizer)

        Returns:
            Predictions
        """
        if self.mitigated_model is None:
            raise RuntimeError("Threshold model not fitted yet")

        if sensitive_features is None:
            raise ValueError(
                "ThresholdOptimizer requires sensitive_features for prediction"
            )

        return self.mitigated_model.predict(X, sensitive_features=sensitive_features)

    def _requires_sensitive_features_for_prediction(self) -> bool:
        """ThresholdOptimizer requires sensitive features for prediction."""
        return True

    def _predict_with_sensitive_features(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with sensitive features for test data."""
        return self.predict(X, sensitive_features=self.sensitive_features_test)
