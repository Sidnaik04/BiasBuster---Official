import pandas as pd
import numpy as np


def encode_features_for_inference(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            # Categorical: encode to numeric codes
            X[col] = (
                X[col]
                .astype(str)
                .fillna("UNKNOWN")
                .astype("category")
                .cat.codes.astype(float)
            )
        else:
            # Numeric columns: coerce and ensure float
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(float)

    # Final safety: ensure ALL columns are float64
    X = X.astype(np.float64)

    # Replace any inf or extreme values
    X = X.replace([np.inf, -np.inf], 0.0)
    X = X.fillna(0.0)

    return X
