import joblib
from pathlib import Path
import sys


# Define FairModelBundle to satisfy pickle loading of custom objects
class FairModelBundle:
    pass


import __main__

if not hasattr(__main__, "FairModelBundle"):
    setattr(__main__, "FairModelBundle", FairModelBundle)

if "__mp_main__" in sys.modules:
    setattr(sys.modules["__mp_main__"], "FairModelBundle", FairModelBundle)

ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}


def load_model(path: str):
    model_path = Path(path)
    if model_path.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_EXTENSIONS))
        raise ValueError(
            f"Unsupported model file type: {model_path.suffix}. Allowed types: {allowed}"
        )

    try:
        obj = joblib.load(path)
        if type(obj).__name__ == "FairModelBundle" and hasattr(obj, "model"):
            return obj.model, getattr(obj, "preprocessor", None)
        return obj, None
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
