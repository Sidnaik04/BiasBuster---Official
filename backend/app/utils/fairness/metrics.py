import numpy as np


def selection_rate(y_pred):
    return float((y_pred == 1).mean())


def true_positive_rate(y_true, y_pred):
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return float((y_pred[positives] == 1).mean())


def demographic_parity_difference(group_rates: dict):
    return max(group_rates.values()) - min(group_rates.values())


def disparate_impact_ratio(group_rates: dict):
    rates = list(group_rates.values())
    return min(rates) / max(rates) if max(rates) > 0 else 0.0


def equal_opportunity_difference(group_tprs: dict):
    return max(group_tprs.values()) - min(group_tprs.values())
