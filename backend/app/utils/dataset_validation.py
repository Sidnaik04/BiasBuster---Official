import pandas as pd


def validate_dataset_health(df: pd.DataFrame) -> str:
    if df.empty:
        raise ValueError("Dataset contains no rows")

    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df.isnull().sum().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "missing_values": missing_values,
        "column_names": df.columns.astype(str).tolist(),
    }
