import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def preprocess_dataset(df: pd.DataFrame, target: str, sensitive: str):

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    if sensitive not in df.columns:
        raise ValueError(f"Sensitive attribute '{sensitive}' not found")

    df = df.copy()

    # -----------------------------
    # Target encoding
    # -----------------------------
    df[target] = df[target].astype(str).str.strip().str.lower()

    binary_map = {
        "<=50k": 0,
        "<=50k.": 0,
        ">50k": 1,
        ">50k.": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "0": 0,
        "1": 1,
    }

    unique_vals = set(df[target].unique())

    if unique_vals.issubset(binary_map.keys()):
        df[target] = df[target].map(binary_map)
    else:
        df[target] = pd.factorize(df[target])[0]

    # -----------------------------
    # Sensitive attribute
    # -----------------------------
    df[sensitive] = df[sensitive].astype(str).str.strip()

    # -----------------------------
    # Feature encoding
    # -----------------------------
    X = df.drop(columns=[target])

    for col in X.columns:
        if is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            continue

        X[col] = X[col].astype("string").fillna("__missing__").str.strip().astype(str)
        X[col] = pd.Categorical(X[col]).codes.astype(np.int64)

    y = df[target]
    sensitive_series = df[sensitive]

    return X, y, sensitive_series


def align_features(df: pd.DataFrame, model) -> pd.DataFrame:
    expected = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "estimator_") and hasattr(
        model.estimator_, "feature_names_in_"
    ):
        expected = list(model.estimator_.feature_names_in_)

    if not expected:
        return df

    if set(df.columns) == set(expected):
        return df[expected]

    import re

    def clean(x):
        return re.sub(r"[^a-zA-Z0-9]", "_", str(x))

    cleaned_expected = {clean(x): x for x in expected}

    new_cols = []
    for c in df.columns:
        cl = clean(c)
        if cl in cleaned_expected:
            new_cols.append(cleaned_expected[cl])
        else:
            new_cols.append(c)

    df_aligned = df.copy()
    df_aligned.columns = new_cols

    available = [c for c in expected if c in df_aligned.columns]
    if len(available) == len(expected):
        return df_aligned[available]
    return df_aligned


def apply_preprocessor(raw_X: pd.DataFrame, preprocessor, df_index) -> pd.DataFrame:
    if not preprocessor:
        return None
    transformed = preprocessor.transform(raw_X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if hasattr(preprocessor, "get_feature_names_out"):
        return pd.DataFrame(
            transformed, columns=preprocessor.get_feature_names_out(), index=df_index
        )
    else:
        return pd.DataFrame(transformed, index=df_index)
