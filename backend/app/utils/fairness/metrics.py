import numpy as np


def selection_rate(y_pred):
    return float((y_pred == 1).mean())


def true_positive_rate(y_true, y_pred):
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return float((y_pred[positives] == 1).mean())


# Backwards-compatible API expected by other modules
def compute_selection_rate(y_pred):
    """Compatibility wrapper for selection rate.

    Some modules import `compute_selection_rate` — provide wrapper.
    """
    return selection_rate(y_pred)


def compute_true_positive_rate(y_true, y_pred):
    """Compatibility wrapper for true positive rate."""
    return true_positive_rate(y_true, y_pred)


def compute_false_positive_rate(y_true, y_pred):
    """Compute false positive rate: FP / N where N = negatives count.

    Returns 0.0 if there are no negative samples.
    """
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    fp = ((y_pred[negatives] == 1)).sum()
    return float(fp / negatives.sum())


def demographic_parity_difference(group_rates: dict):
    return max(group_rates.values()) - min(group_rates.values())


def disparate_impact_ratio(group_rates: dict):
    rates = list(group_rates.values())
    return min(rates) / max(rates) if max(rates) > 0 else 0.0


def equal_opportunity_difference(group_tprs: dict):
    return max(group_tprs.values()) - min(group_tprs.values())
