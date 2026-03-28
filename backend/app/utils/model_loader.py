from pathlib import Path
import joblib
from app.config import settings

MODEL_DIR = Path(settings.TEMP_DIR) / "models"


def load_model(filename: str):
    path = MODEL_DIR / filename

    if not path.exists():
        raise ValueError("Model file not found")

    try:
        return joblib.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
