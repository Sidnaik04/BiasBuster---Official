import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")
