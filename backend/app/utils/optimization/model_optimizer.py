"""
Model Optimization Assistant: Hyperparameter tuning with fairness awareness.

Combines accuracy and fairness objectives using:
1. GridSearchCV: exhaustive parameter search
2. Optuna: probabilistic optimization
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
import traceback

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from app.utils.fairness_metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
)


class FairnessAwareOptimizer:
    """
    Optimizer that balances accuracy and fairness metrics.

    Combined score formula:
    combined_score = (0.6 * accuracy) + (0.4 * fairness_score)

    Where:
    fairness_score = 1 - (|DPD| + |EOD|) / 2
    """

    def __init__(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
        accuracy_weight: float = 0.6,
        fairness_weight: float = 0.4,
    ):
        """
        Initialize optimizer.

        Args:
            model: sklearn-compatible classifier
            X: Feature matrix
            y: Target labels
            sensitive_features: DataFrame with sensitive attributes
            test_size: Proportion for test split
            random_state: Random seed
            accuracy_weight: Weight for accuracy in combined score (0-1)
            fairness_weight: Weight for fairness in combined score (0-1)
        """
        self.model = model
        self.X = X
        self.y = y
        self.sensitive_features = sensitive_features
        self.test_size = test_size
        self.random_state = random_state
        self.accuracy_weight = accuracy_weight
        self.fairness_weight = fairness_weight

        # Verify weights sum to 1
        if not np.isclose(accuracy_weight + fairness_weight, 1.0):
            raise ValueError("accuracy_weight + fairness_weight must equal 1.0")

        # Split data (stratified to preserve class distribution)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Also split sensitive features
        self.sensitive_train, self.sensitive_test = train_test_split(
            sensitive_features,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        self.trial_history = []

    def _compute_fairness_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute fairness score based on DPD and EOD.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values

        Returns:
            (fairness_score, metrics_dict)
        """

        # Handle edge case: all same class in predictions
        if len(np.unique(y_pred)) < 2:
            return 0.0, {"dpd": 1.0, "eod": 1.0, "status": "degenerate"}

        try:
            # Group by sensitive attribute
            groups = np.unique(sensitive_attr)

            if len(groups) < 2:
                # Only one group - no fairness concerns
                return 1.0, {"dpd": 0.0, "eod": 0.0, "status": "single_group"}

            # Compute selection rates per group (DPD)
            dpd_rates = {}
            for group in groups:
                mask = sensitive_attr == group
                if mask.sum() > 0:
                    dpd_rates[group] = (y_pred[mask] == 1).mean()

            # Compute TPRs per group (EOD)
            eod_rates = {}
            for group in groups:
                mask = (sensitive_attr == group) & (y_true == 1)
                if mask.sum() > 0:
                    eod_rates[group] = (y_pred[mask] == 1).mean()

            # Compute DPD and EOD
            dpd = demographic_parity_difference(list(dpd_rates.values()))
            eod = equal_opportunity_difference(list(eod_rates.values()))

            # Fairness score: 1 - (|DPD| + |EOD|) / 2
            # Higher is better (1.0 = perfectly fair)
            fairness_score = 1.0 - (abs(dpd) + abs(eod)) / 2

            # Clamp to [0, 1]
            fairness_score = np.clip(fairness_score, 0.0, 1.0)

            return fairness_score, {
                "dpd": round(dpd, 4),
                "eod": round(eod, 4),
                "status": "computed",
            }

        except Exception as e:
            # Fallback: if fairness computation fails, return neutral score
            print(f"Warning: Fairness computation failed: {str(e)}")
            return 0.5, {"dpd": None, "eod": None, "status": f"error: {str(e)}"}

    def _compute_combined_score(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute combined accuracy + fairness score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attr: Sensitive attribute values

        Returns:
            (combined_score, metrics_dict)
        """

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Fairness
        fairness_score, fairness_metrics = self._compute_fairness_score(
            y_true, y_pred, sensitive_attr
        )

        # Combined (weighted)
        combined = (
            self.accuracy_weight * accuracy + self.fairness_weight * fairness_score
        )

        metrics = {
            "accuracy": round(accuracy, 4),
            "fairness_score": round(fairness_score, 4),
            "combined_score": round(combined, 4),
            **fairness_metrics,
        }

        return combined, metrics

    def optimize_with_gridsearch(
        self, param_grid: Dict[str, List[Any]], cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize using GridSearchCV.

        Args:
            param_grid: Parameter grid for GridSearchCV
            cv: Number of cross-validation folds

        Returns:
            Optimization results dictionary
        """

        print(f"Starting GridSearchCV with {len(param_grid)} parameter combinations...")

        # Handle Pipeline: add step name prefix to parameter names
        if hasattr(self.model, "steps"):
            step_name = self.model.steps[-1][0]
            param_grid = {
                f"{step_name}__{key}": value for key, value in param_grid.items()
            }

        # GridSearch with accuracy scorer
        # Note: We use accuracy for CV, then evaluate fairness on test set
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            ),
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(self.X_train, self.y_train)

        # Get best model predictions on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        sensitive_attr = self.sensitive_test.iloc[:, 0].values

        _, test_metrics = self._compute_combined_score(
            self.y_test, y_pred, sensitive_attr
        )

        # Collect trial history
        self.trial_history = [
            {
                "params": dict(params),
                "cv_score": float(score),
                "mean_test_accuracy": round(
                    accuracy_score(self.y_test, best_model.predict(self.X_test)), 4
                ),
                "fairness_score": test_metrics.get("fairness_score", 0.0),
                "combined_score": test_metrics.get("combined_score", 0.0),
            }
            for params, score in zip(
                grid_search.cv_results_["params"],
                grid_search.cv_results_["mean_test_score"],
            )
        ]

        return {
            "best_params": dict(grid_search.best_params_),
            "best_score": float(grid_search.best_score_),
            "accuracy": test_metrics["accuracy"],
            "fairness_score": test_metrics["fairness_score"],
            "combined_score": test_metrics["combined_score"],
            "dpd": test_metrics.get("dpd"),
            "eod": test_metrics.get("eod"),
            "optimization_method": "gridsearch",
            "trials_run": len(grid_search.cv_results_["params"]),
            "comparison": self.trial_history[:10],  # Top 10 trials
        }

    def optimize_with_optuna(
        self,
        param_distributions: Dict[str, Any],
        n_trials: int = 20,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """
        Optimize using Optuna.

        Args:
            param_distributions: Parameter distributions for sampling
            n_trials: Number of trials
            timeout: Max time in seconds

        Returns:
            Optimization results dictionary
        """

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        print(f"Starting Optuna optimization with {n_trials} trials...")

        def objective(trial):
            """Optuna objective function."""
            try:
                # Sample hyperparameters
                params = {}
                for param_name, param_spec in param_distributions.items():
                    if isinstance(param_spec, dict):
                        if param_spec.get("type") == "int":
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_spec.get("low", 1),
                                param_spec.get("high", 10),
                            )
                        elif param_spec.get("type") == "float":
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_spec.get("low", 0.0),
                                param_spec.get("high", 1.0),
                            )
                        elif param_spec.get("type") == "categorical":
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_spec.get("choices", [])
                            )
                    else:
                        # Assume it's a list of choices
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_spec
                        )

                # Clone and set params
                from sklearn.base import clone

                model = clone(self.model)

                # Handle Pipeline: add step name prefix to parameter names
                params_to_set = params.copy()
                if hasattr(model, "steps"):
                    step_name = model.steps[-1][0]
                    params_to_set = {
                        f"{step_name}__{key}": value for key, value in params.items()
                    }

                model.set_params(**params_to_set)

                # Train on training set
                model.fit(self.X_train, self.y_train)

                # Evaluate on test set
                y_pred = model.predict(self.X_test)
                sensitive_attr = self.sensitive_test.iloc[:, 0].values

                # Compute combined score
                combined, metrics = self._compute_combined_score(
                    self.y_test, y_pred, sensitive_attr
                )

                # Store trial history
                self.trial_history.append(
                    {
                        "trial_id": trial.number,
                        "params": dict(params),
                        "accuracy": metrics["accuracy"],
                        "fairness_score": metrics["fairness_score"],
                        "combined_score": metrics["combined_score"],
                        "dpd": metrics.get("dpd"),
                        "eod": metrics.get("eod"),
                    }
                )

                return combined

            except Exception as e:
                print(f"Trial {trial.number} failed: {str(e)}")
                return 0.0

        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner()

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Get best trial
        best_trial = study.best_trial

        # Get metrics from trial history (best trial by combined score)
        best_trial_metrics = {
            "accuracy": 0.0,
            "fairness_score": 0.0,
            "combined_score": 0.0,
            "dpd": None,
            "eod": None,
        }
        if self.trial_history:
            best_by_score = max(
                self.trial_history, key=lambda x: x.get("combined_score", 0.0)
            )
            best_trial_metrics.update(best_by_score)

        return {
            "best_params": dict(best_trial.params) if best_trial else {},
            "best_score": (
                float(best_trial.value) if best_trial and best_trial.value else 0.0
            ),
            "accuracy": best_trial_metrics.get("accuracy", 0.0),
            "fairness_score": best_trial_metrics.get("fairness_score", 0.0),
            "combined_score": best_trial_metrics.get("combined_score", 0.0),
            "dpd": best_trial_metrics.get("dpd"),
            "eod": best_trial_metrics.get("eod"),
            "optimization_method": "optuna",
            "trials_run": len(study.trials),
            "comparison": sorted(
                self.trial_history,
                key=lambda x: x.get("combined_score", 0.0),
                reverse=True,
            )[
                :10
            ],  # Top 10 trials
        }


# Default parameter grids for common models
PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [100, 200, 500],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "DecisionTreeClassifier": {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy"],
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
    },
}

# Default parameter distributions for Optuna
OPTUNA_PARAMS = {
    "LogisticRegression": {
        "C": {"type": "float", "low": 0.001, "high": 100},
        "penalty": {"type": "categorical", "choices": ["l2"]},
        "solver": {"type": "categorical", "choices": ["lbfgs", "liblinear"]},
        "max_iter": {"type": "int", "low": 100, "high": 500},
    },
    "RandomForestClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    },
    "DecisionTreeClassifier": {
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
    },
    "SVC": {
        "C": {"type": "float", "low": 0.1, "high": 100},
        "kernel": {"type": "categorical", "choices": ["linear", "rbf"]},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
    },
    "GradientBoostingClassifier": {
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "learning_rate": {"type": "float", "low": 0.001, "high": 0.1},
        "max_depth": {"type": "int", "low": 3, "high": 10},
    },
}


def extract_base_estimator(model):
    """
    Extract base estimator from Pipeline or return model as-is.

    For sklearn Pipelines, extracts the final estimator.
    For other models, returns the model unchanged.

    Args:
        model: sklearn model or Pipeline

    Returns:
        Base estimator
    """
    if hasattr(model, "steps"):
        # It's a Pipeline - get the last estimator
        return model.steps[-1][1]
    elif hasattr(model, "named_steps"):
        # Alternative Pipeline format
        return list(model.named_steps.values())[-1]
    return model


def get_model_class_name(model) -> str:
    """
    Get the class name of the model, handling Pipelines.

    Args:
        model: sklearn model or Pipeline

    Returns:
        Class name of the base estimator
    """
    base_estimator = extract_base_estimator(model)
    return base_estimator.__class__.__name__


def get_optuna_params(model_class_name: str) -> Optional[Dict[str, Any]]:
    """Get default Optuna parameter distributions for a model."""
    return OPTUNA_PARAMS.get(model_class_name)


def _add_pipeline_prefix(params: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Add Pipeline step prefix to parameter names if needed.

    For Pipelines, sklearn requires prefixed parameter names like 'estimator__C'.
    For direct models, no prefix is needed.

    Args:
        params: Original parameter dictionary
        model: sklearn model or Pipeline

    Returns:
        Dictionary with properly prefixed parameters
    """
    if not hasattr(model, "steps"):
        # Not a Pipeline, return as-is
        return params

    # Get the name of the last step (the estimator)
    step_name = model.steps[-1][0]

    # Prefix all parameters with step name
    return {f"{step_name}__{key}": value for key, value in params.items()}


def get_param_grid(model_class_name: str) -> Optional[Dict[str, List[Any]]]:
    """Get default parameter grid for a model."""
    return PARAM_GRIDS.get(model_class_name)


def get_optuna_params(model_class_name: str) -> Optional[Dict[str, Any]]:
    """Get default Optuna parameter distributions for a model."""
    return OPTUNA_PARAMS.get(model_class_name)
