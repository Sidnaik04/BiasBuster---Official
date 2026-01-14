from pathlib import Path
import joblib
from typing import Any, Dict

from sklearn.base import BaseEstimator
from fairlearn.postprocessing import ThresholdOptimizer

ALLOWED_METHODS = [
    "predict",
    "predict_proba",
    "predict_score",
    "predict_label",
    "transform",
]

def has_allowed_method(obj: Any) -> bool:
    return any(hasattr(obj, m) for m in ALLOWED_METHODS)

def extract_model_from_dict(bundle: Dict) -> Any:
    # Common patterns:
    # { "model": estimator, "threshold": 0.6 }
    # { "estimator": estimator }
    for key in ["model", "estimator", "base_model", "clf"]:
        if key in bundle:
            return bundle[key]
    return None

def extract_wrapped_model(obj: Any) -> Any:
    possible_attrs = ["model", "base_model", "estimator", "clf", "inner_model"]
    for attr in possible_attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return None

def safe_load_model_from_path(path: Path) -> Dict:
    try:
        model_obj = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            "Model file could not be loaded. Ensure it's a joblib/pickle file."
        ) from exc

    if isinstance(model_obj, ThresholdOptimizer):
        base = model_obj.estimator_
        if not has_allowed_method(base):
            raise ValueError(
                "ThresholdOptimizer wrapper loaded, but inner estimator lacks prediction methods."
            )
        return {
            "model": model_obj,
            "model_type": "ThresholdOptimizer",
            "supports_proba": hasattr(base, "predict_proba"),
        }

    if isinstance(model_obj, dict):
        inner = extract_model_from_dict(model_obj)
        if inner is None:
            raise ValueError(
                "Model is a dict but does not contain a valid underlying model (e.g., 'model' key)."
            )
        if not has_allowed_method(inner):
            raise ValueError("Underlying model in dict does not support prediction.")
        return {
            "model": model_obj,
            "model_type": f"DictBundle({type(inner).__name__})",
            "supports_proba": hasattr(inner, "predict_proba"),
        }

    if isinstance(model_obj, BaseEstimator):
        if not has_allowed_method(model_obj):
            raise ValueError("Uploaded sklearn estimator does not support prediction.")
        return {
            "model": model_obj,
            "model_type": type(model_obj).__name__,
            "supports_proba": hasattr(model_obj, "predict_proba"),
        }

    wrapped = extract_wrapped_model(model_obj)
    if wrapped is not None and has_allowed_method(wrapped):
        return {
            "model": model_obj,
            "model_type": f"Wrapper({type(wrapped).__name__})",
            "supports_proba": hasattr(wrapped, "predict_proba"),
        }

    if has_allowed_method(model_obj):
        return {
            "model": model_obj,
            "model_type": type(model_obj).__name__,
            "supports_proba": hasattr(model_obj, "predict_proba"),
        }

    raise ValueError(
        f"Uploaded object of type '{type(model_obj).__name__}' "
        f"is not a valid predictive model or wrapper."
    )
