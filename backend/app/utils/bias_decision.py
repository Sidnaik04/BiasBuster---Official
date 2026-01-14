def evaluate_bias(dpd, eod, dir_ratio):
    violations = {
        "dpd": abs(dpd) > 0.10,
        "eod": abs(eod) > 0.10,
        "dir": dir_ratio < 0.80,
    }

    bias_present = any(violations.values())

    severity = (
        violations["dpd"] * 0.33 + violations["eod"] * 0.33 + violations["dir"] * 0.34
    )

    return {
        "bias_present": bias_present,
        "violations": violations,
        "severity_score": round(severity * 10, 2),
    }
