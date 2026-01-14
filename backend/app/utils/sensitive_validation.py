import pandas as pd


def validate_sensitive_columns(df: pd.DataFrame, sensitive_columns: list[str]) -> dict:
    missing = [c for c in sensitive_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Sensitive columns missing: {missing}")

    audit = {}

    for col in sensitive_columns:
        unique_vals = df[col].dropna().unique()

        if len(unique_vals) < 2:
            raise ValueError(
                f"Sensitive column '{col}'  has fewer than 2 unique values"
            )

        audit[col] = {
            "unique_gropus": int(len(unique_vals)),
            "groups": [str(v) for v in unique_vals],
        }

    return audit
