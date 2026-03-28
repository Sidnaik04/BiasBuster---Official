import pandas as pd
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer


def predict_labels(model, X, sensitive_features=None):
    if isinstance(model, ThresholdOptimizer):
        if sensitive_features is None:
            raise ValueError(
                "ThresholdOptimizer requires sensitive_features for prediction. "
                "Please re-run bias detection using the base model, "
                "or provide sensitive features explicitly."
            )
        return model.predict(X, sensitive_features=sensitive_features)

    if hasattr(model, "predict"):
        return model.predict(X)

    raise ValueError("Model does not support prediction")


def predict_proba_positive(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]

    return None
