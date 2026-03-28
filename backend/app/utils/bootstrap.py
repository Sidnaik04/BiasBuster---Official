import numpy as np

import numpy as np


def bootstrap_ci(values, n_bootstrap=100, ci=95):
    """
    Generic bootstrap confidence interval.
    values: list or array of metric values
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return None

    stats = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        stats.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(stats, alpha)
    upper = np.percentile(stats, 100 - alpha)

    return round(lower, 4), round(upper, 4)


import numpy as np


def bootstrap_ci(values, n_bootstrap=100, ci=95):
    """
    Generic bootstrap confidence interval.
    values: list or array of metric values
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return None

    stats = []
    n = len(values)

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        stats.append(np.mean(sample))

    alpha = (100 - ci) / 2
    lower = np.percentile(stats, alpha)
    upper = np.percentile(stats, 100 - alpha)

    return round(lower, 4), round(upper, 4)
