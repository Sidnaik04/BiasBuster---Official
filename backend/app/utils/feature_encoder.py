import pandas as pd
import numpy as np


def encode_features_for_inference(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            X[col] = (
                X[col]
                .astype(str)
                .fillna("UNKNOWN")
                .astype("category")
                .cat.codes.astype(np.int64)
            )
        else:
            # numeric columns: coerce safely
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X
