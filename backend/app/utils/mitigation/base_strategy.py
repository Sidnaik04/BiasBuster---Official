"""Base strategy interface for unified mitigation orchestration."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class BaseMitigationStrategy(ABC):
    """Abstract base class for all mitigation strategies."""

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_features_train: pd.DataFrame,
        sensitive_features_test: pd.DataFrame,
        target_column: str,
        strategy_name: str,
    ):
        """
        Initialize strategy with standardized inputs.

        Args:
            model: sklearn model or Pipeline
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels (ground truth)
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing
            target_column: Target column name
            strategy_name: Name of the strategy
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features_train = sensitive_features_train
        self.sensitive_features_test = sensitive_features_test
        self.target_column = target_column
        self.strategy_name = strategy_name

        # Results will be populated during fit
        self.mitigated_model = None
        self.X_train_transformed = None
        self.y_train_transformed = None
        self.diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def fit(self) -> bool:
        """
        Fit the mitigation strategy on training data.

        Returns:
            True if successful, False if failed
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using mitigated model.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.mitigated_model is None:
            raise RuntimeError(f"{self.strategy_name}: Model not fitted yet")

        # Check if model requires sensitive features for prediction
        if self._requires_sensitive_features_for_prediction():
            # This should be overridden in subclasses that need it
            return self._predict_with_sensitive_features(X)
        else:
            return self.mitigated_model.predict(X)

    def _requires_sensitive_features_for_prediction(self) -> bool:
        """Check if strategy requires sensitive features for prediction."""
        return False

    def _predict_with_sensitive_features(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with sensitive features (for strategies that need them)."""
        raise NotImplementedError(
            f"{self.strategy_name} requires sensitive features for prediction"
        )

    def get_strategy_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the strategy.

        Returns:
            Metadata dictionary
        """
        return {
            "strategy_name": self.strategy_name,
            "model_type": type(self.model).__name__,
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "n_sensitive_features": (
                self.sensitive_features_train.shape[1]
                if hasattr(self.sensitive_features_train, "shape")
                else 1
            ),
        }

    def get_execution_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about strategy execution.

        Returns:
            Diagnostics dictionary
        """
        return {
            "strategy": self.strategy_name,
            "train_shape_before": self.X_train.shape,
            "train_shape_after": (
                self.X_train_transformed.shape
                if self.X_train_transformed is not None
                else None
            ),
            "y_train_shape_before": self.y_train.shape,
            "y_train_shape_after": (
                self.y_train_transformed.shape
                if self.y_train_transformed is not None
                else None
            ),
            "sensitive_features_shape": (
                self.sensitive_features_train.shape
                if hasattr(self.sensitive_features_train, "shape")
                else len(self.sensitive_features_train)
            ),
            **self.diagnostics,
        }

    @staticmethod
    def validate_inputs(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sensitive_features_train: pd.DataFrame,
        sensitive_features_test: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input data.

        Args:
            model: sklearn model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features_train: Sensitive features for training
            sensitive_features_test: Sensitive features for testing

        Returns:
            (is_valid, error_message)
        """
        if model is None:
            return False, "Model is None"

        if X_train is None or len(X_train) == 0:
            return False, "X_train is empty"

        if y_train is None or len(y_train) == 0:
            return False, "y_train is empty"

        if X_test is None or len(X_test) == 0:
            return False, "X_test is empty"

        if y_test is None or len(y_test) == 0:
            return False, "y_test is empty"

        if len(X_train) != len(y_train):
            return False, "X_train and y_train length mismatch"

        if len(X_test) != len(y_test):
            return False, "X_test and y_test length mismatch"

        if len(X_train) != len(sensitive_features_train):
            return False, "X_train and sensitive_features_train length mismatch"

        if len(X_test) != len(sensitive_features_test):
            return False, "X_test and sensitive_features_test length mismatch"

        if sensitive_features_train is None or len(sensitive_features_train) == 0:
            return False, "sensitive_features_train is empty"

        if sensitive_features_test is None or len(sensitive_features_test) == 0:
            return False, "sensitive_features_test is empty"

        return True, None
