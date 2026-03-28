import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.utils.fairness.metrics import (
    selection_rate,
    true_positive_rate,
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
)


def evaluate_baseline(y_true, y_pred, sensitive):
    unique_labels = set(pd.Series(y_true).dropna().unique()) | set(
        pd.Series(y_pred).dropna().unique()
    )
    metric_average = "binary" if len(unique_labels) <= 2 else "macro"

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "sensitive": sensitive})

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
