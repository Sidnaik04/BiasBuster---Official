def compare_fairness(before: dict, after: dict) -> dict:
    """
    Compare before and after fairness metrics for a single sensitive attribute.
    """

    improvements = {}

    for metric in ["dpd", "eod", "dir", "severity_score"]:
        before_val = before.get(metric)
        after_val = after.get(metric)

        if before_val is None or after_val is None:
            continue

        if metric == "dir":
            # DIR closer to 1 is better
            improved = abs(1 - after_val) < abs(1 - before_val)
        else:
            # DPD, EOD, severity closer to 0 is better
            improved = abs(after_val) < abs(before_val)

        improvements[metric] = {
            "before": before_val,
            "after": after_val,
            "improved": improved,
        }

    fairness_improved = any(v["improved"] for v in improvements.values())

    return {
        "metric_comparison": improvements,
        "fairness_improved": fairness_improved,
        "note": (
            "Fairness improvement is metric-dependent and does not imply complete bias removal."
        ),
    }
