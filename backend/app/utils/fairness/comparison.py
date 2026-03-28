def compare_metrics(before: dict, after: dict):
    """
    Compare baseline vs mitigated fairness + performance.
    """

    fairness_before = before["fairness"]
    fairness_after = after["fairness"]

    perf_before = before["performance"]
    perf_after = after["performance"]

    report = {}

    # -------------------------------------------------
    # Fairness comparison
    # -------------------------------------------------

    dpd_before = fairness_before["dpd"]
    dpd_after = fairness_after["dpd"]

    eod_before = fairness_before["eod"]
    eod_after = fairness_after["eod"]

    dir_before = fairness_before["dir"]
    dir_after = fairness_after["dir"]

    report["fairness_changes"] = {
        "dpd": {
            "before": dpd_before,
            "after": dpd_after,
            "improved": abs(dpd_after) < abs(dpd_before),
        },
        "eod": {
            "before": eod_before,
            "after": eod_after,
            "improved": abs(eod_after) < abs(eod_before),
        },
        "dir": {
            "before": dir_before,
            "after": dir_after,
            "improved": abs(1 - dir_after) < abs(1 - dir_before),
        },
    }

    # -------------------------------------------------
    # Performance comparison
    # -------------------------------------------------

    accuracy_before = perf_before["accuracy"]
    accuracy_after = perf_after["accuracy"]

    precision_before = perf_before["precision"]
    precision_after = perf_after["precision"]

    recall_before = perf_before["recall"]
    recall_after = perf_after["recall"]

    f1_before = perf_before["f1"]
    f1_after = perf_after["f1"]

    report["performance_changes"] = {
        "accuracy": {
            "before": accuracy_before,
            "after": accuracy_after,
            "change": accuracy_after - accuracy_before,
        },
        "precision": {
            "before": precision_before,
            "after": precision_after,
            "change": precision_after - precision_before,
        },
        "recall": {
            "before": recall_before,
            "after": recall_after,
            "change": recall_after - recall_before,
        },
        "f1": {
            "before": f1_before,
            "after": f1_after,
            "change": f1_after - f1_before,
        },
    }

    # -------------------------------------------------
    # Overall fairness improvement
    # -------------------------------------------------

    improvements = [
        report["fairness_changes"]["dpd"]["improved"],
        report["fairness_changes"]["eod"]["improved"],
        report["fairness_changes"]["dir"]["improved"],
    ]

    fairness_improved = sum(improvements) >= 2

    report["overall_assessment"] = {
        "fairness_improved": fairness_improved,
        "accuracy_drop": accuracy_after < accuracy_before,
    }

    return report
