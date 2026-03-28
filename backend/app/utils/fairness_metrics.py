import numpy as np
import pandas as pd


def _safe_array(arr):
    return np.asarray(arr, dtype=float)


def selection_rate(y_pred):
    y_pred = _safe_array(y_pred)
    return float(np.mean(y_pred == 1))


def true_positive_rate(y_true, y_pred):
    y_true = _safe_array(y_true)
    y_pred = _safe_array(y_pred)

    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0

    return float((y_pred[positives] == 1).mean())


def demographic_parity_difference(group_rates: dict) -> float:
    return max(group_rates.values()) - min(group_rates.values())


def equal_opportunity_difference(group_tprs: dict) -> float:
    return max(group_tprs.values()) - min(group_tprs.values())


def disparate_impact_ratio(group_rates: dict) -> float:
    rates = list(group_rates.values())
    min_rate = min(rates)
    max_rate = max(rates)
    if max_rate == 0:
        return 0.0
    return min_rate / max_rate
