import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.utils.fairness.metrics import (
    selection_rate,
    true_positive_rate,
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
)


def _validate_metric_inputs(y_true, y_pred, sensitive):
    if y_true is None or y_pred is None or sensitive is None:
        raise ValueError("y_true, y_pred, and sensitive must all be provided")

    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    sensitive_s = pd.Series(sensitive)

    if len(y_true_s) == 0 or len(y_pred_s) == 0 or len(sensitive_s) == 0:
        raise ValueError("Metric inputs cannot be empty")
    if not (len(y_true_s) == len(y_pred_s) == len(sensitive_s)):
        raise ValueError(
            f"Metric input length mismatch: y_true={len(y_true_s)}, y_pred={len(y_pred_s)}, sensitive={len(sensitive_s)}"
        )

    y_true_unique = y_true_s.dropna().unique()
    sensitive_unique = sensitive_s.dropna().unique()

    if len(y_true_unique) < 2:
        raise ValueError("y_true must contain at least two classes")
    if len(sensitive_unique) < 2:
        raise ValueError("sensitive must contain at least two groups")

    return y_true_s, y_pred_s, sensitive_s


def evaluate_baseline(y_true, y_pred, sensitive):
    y_true_s, y_pred_s, sensitive_s = _validate_metric_inputs(y_true, y_pred, sensitive)

    unique_labels = set(y_true_s.dropna().unique()) | set(y_pred_s.dropna().unique())
    metric_average = "binary" if len(unique_labels) <= 2 else "macro"

    df = pd.DataFrame(
        {"y_true": y_true_s, "y_pred": y_pred_s, "sensitive": sensitive_s}
    )

    group_rates = {}
    group_tprs = {}

    for group, gdf in df.groupby("sensitive"):
        group_rates[group] = selection_rate(gdf["y_pred"])
        group_tprs[group] = true_positive_rate(gdf["y_true"], gdf["y_pred"])

    fairness = {
        "selection_rate": group_rates,
        "dpd": demographic_parity_difference(group_rates),
        "dir": disparate_impact_ratio(group_rates),
        "eod": equal_opportunity_difference(group_tprs),
    }

    performance = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average=metric_average,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average=metric_average,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average=metric_average,
            zero_division=0,
        ),
    }

    return {
        "performance": performance,
        "fairness": fairness,
    }
