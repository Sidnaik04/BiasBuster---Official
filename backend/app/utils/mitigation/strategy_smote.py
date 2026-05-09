"""SMOTE strategy implementation."""

import logging
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone

from .base_strategy import BaseMitigationStrategy

logger = logging.getLogger(__name__)


class SMOTEStrategy(BaseMitigationStrategy):
    """SMOTE (Synthetic Minority Over-sampling Technique) strategy."""

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
        random_state: int = 42,
    ):
        """
        Initialize SMOTE strategy.

        Args:
            model: Base sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
            random_state: Random state for reproducibility
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
            strategy_name="smote",
        )
        self.random_state = random_state

    def fit(self) -> bool:
        """
        Fit model on SMOTE-resampled training data.

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
                logger.error(f"SMOTE strategy validation failed: {error_msg}")
                self.diagnostics["error"] = error_msg
                return False

            # Convert X_train to numpy for SMOTE
            if isinstance(self.X_train, pd.DataFrame):
                X_train_array = self.X_train.values
                feature_names = self.X_train.columns.tolist()
            else:
                X_train_array = np.asarray(self.X_train)
                feature_names = None

            # Convert y_train to numpy
            if isinstance(self.y_train, pd.Series):
                y_train_array = self.y_train.values
            else:
                y_train_array = np.asarray(self.y_train)

            self.diagnostics["train_shape_before"] = X_train_array.shape
            self.diagnostics["class_distribution_before"] = (
                self._get_class_distribution(y_train_array)
            )

            # Apply SMOTE
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_train_array, y_train_array)

            # Recreate DataFrame if original was DataFrame
            if feature_names:
                X_resampled = pd.DataFrame(X_resampled, columns=feature_names)

            self.diagnostics["train_shape_after"] = X_resampled.shape
            self.diagnostics["class_distribution_after"] = self._get_class_distribution(
                y_resampled
            )

            logger.info(
                f"SMOTE: Resampled data from {X_train_array.shape[0]} to {X_resampled.shape[0]} samples"
            )

            # Fit model on resampled data
            mitigated_model = clone(self.model)
            mitigated_model.fit(X_resampled, y_resampled)

            self.mitigated_model = mitigated_model
            self.diagnostics["fit_success"] = True
            logger.info("SMOTE strategy fitted successfully")
            return True

        except Exception as e:
            logger.error(f"SMOTE strategy fit failed: {str(e)}")
            self.diagnostics["error"] = str(e)
            return False

    def predict(self, X: np.ndarray, sensitive_features=None) -> np.ndarray:
        """
        Make predictions using SMOTE-trained model.

        Args:
            X: Feature matrix
            sensitive_features: Sensitive features (not used for SMOTE prediction)

        Returns:
            Predictions
        """
        if self.mitigated_model is None:
            raise RuntimeError("SMOTE model not fitted yet")

        return self.mitigated_model.predict(X)

    def _requires_sensitive_features_for_prediction(self) -> bool:
        """SMOTE doesn't require sensitive features for prediction."""
        return False

    @staticmethod
    def _get_class_distribution(y):
        """Get class distribution as dictionary."""
        unique, counts = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
