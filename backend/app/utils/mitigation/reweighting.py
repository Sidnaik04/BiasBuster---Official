def compute_sample_weights(sensitive_series):
    """
    Assign inverse-frequency weights to sensitive groups.
    """
    counts = sensitive_series.value_counts()
    total = len(sensitive_series)

    weights = sensitive_series.map(
        lambda x: total / (counts[x] * len(counts))
    )

    return weights