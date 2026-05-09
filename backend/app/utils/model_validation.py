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


def _get_final_estimator(obj: Any) -> Any:
    """Return the final estimator inside a pipeline-like object.

    If the object is a sklearn Pipeline (or pipeline-like with `steps` or `named_steps`),
    this will recurse into the last step. Otherwise returns the object itself.
    """
    try:
        steps = getattr(obj, "steps", None)
        if isinstance(steps, list) and len(steps) > 0:
            final = steps[-1][1]
            return _get_final_estimator(final)

        named = getattr(obj, "named_steps", None)
        if isinstance(named, dict) and len(named) > 0:
            final = list(named.values())[-1]
            return _get_final_estimator(final)
    except Exception:
        pass
    return obj


def detect_model_type(model: Any) -> str:
    """Detect and return the model type name.

    - If `model` is a Pipeline (or pipeline-like), returns the final estimator's class name.
    - Handles nested pipelines.
    - Falls back to the object's class name for empty/invalid pipelines or unknown structures.
    """
    try:
        final = _get_final_estimator(model)
        return type(final).__name__
    except Exception:
        return type(model).__name__


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
        final_inner = _get_final_estimator(inner)
        return {
            "model": model_obj,
            "model_type": f"DictBundle({detect_model_type(inner)})",
            "supports_proba": hasattr(final_inner, "predict_proba"),
        }

    if isinstance(model_obj, BaseEstimator):
        if not has_allowed_method(model_obj):
            raise ValueError("Uploaded sklearn estimator does not support prediction.")
        final = _get_final_estimator(model_obj)
        return {
            "model": model_obj,
            "model_type": detect_model_type(model_obj),
            "supports_proba": hasattr(final, "predict_proba"),
        }

    wrapped = extract_wrapped_model(model_obj)
    if wrapped is not None and has_allowed_method(wrapped):
        final_wrapped = _get_final_estimator(wrapped)
        return {
            "model": model_obj,
            "model_type": (
                detect_model_type(wrapped)
                if detect_model_type(wrapped)
                else f"Wrapper({type(wrapped).__name__})"
            ),
            "supports_proba": hasattr(final_wrapped, "predict_proba"),
        }

    if has_allowed_method(model_obj):
        final = _get_final_estimator(model_obj)
        return {
            "model": model_obj,
            "model_type": detect_model_type(model_obj),
            "supports_proba": hasattr(final, "predict_proba"),
        }

    raise ValueError(
        f"Uploaded object of type '{type(model_obj).__name__}' "
        f"is not a valid predictive model or wrapper."
    )
