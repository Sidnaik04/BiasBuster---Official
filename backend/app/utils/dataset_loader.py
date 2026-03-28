from pathlib import Path
import pandas as pd
from app.config import settings

DATASET_DIR = Path(settings.TEMP_DIR) / "datasets"


def load_dataset(filename: str) -> pd.DataFrame:
    path = DATASET_DIR / filename

    if not path.exists():
        raise ValueError("Dataset file not found")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load_dataset: {e}")

    return df
