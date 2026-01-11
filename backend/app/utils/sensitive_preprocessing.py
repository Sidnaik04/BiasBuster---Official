import pandas as pd


def bin_age_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df

    if not pd.api.types.is_numeric_dtype(df[col]):
        return df  # don't bin non-numeric age

    df = df.copy()
    df[col + "_group"] = pd.cut(
        df[col],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        right=True,
        include_lowest=True,
    )

    return df
