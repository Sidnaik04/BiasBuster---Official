import pandas as pd

BINARY_MAP = {
    "normal": 0,
    "abnormal": 1,
    "no": 0,
    "yes": 1,
    "false": 0,
    "true": 1,
    0: 0,
    1: 1,
}

DROP_VALUES = {"inconclusive", "unknown", "na", "n/a", ""}


def normalize_value(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


def encode_target_column(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, dict]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # normalize
    df = df.copy()
    df[target_col] = df[target_col].apply(normalize_value)

    # drop inconclusive rows
    before_rows = df.shape[0]
    df = df[~df[target_col].isin(DROP_VALUES)]
    dropped_rows = before_rows - df.shape[0]

    unique_vals = set(df[target_col].dropna().unique())

    # binary case
    if unique_vals.issubset(set(BINARY_MAP.keys())):
        df[target_col] = df[target_col].map(BINARY_MAP)
        audit_mode = "binary"

    # limited non-binary (<= 3 classes)
    elif len(unique_vals) <= 3:
        class_map = {v: i for i, v in enumerate(sorted(unique_vals))}
        df[target_col] = df[target_col].map(class_map)
        audit_mode = "multiclass"

    else:
        raise ValueError(
            f"Target column has {len(unique_vals)} unique values"
            "fairness audit supports binary to small multiclass targets only."
        )

    return df, {
        "audit_mode": audit_mode,
        "dropped_rows": dropped_rows,
        "unique_classes": len(unique_vals),
    }
