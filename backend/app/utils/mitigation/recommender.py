def recommend_strategy(audit_entry: dict) -> dict:
    violations = audit_entry["violations"]
    severity = audit_entry["severity_score"]

    if violations["dir"]:
        strategy = "smote"
        explanation = "Representation imbalance detected (DIR violation)."

    elif violations["dpd"]:
        strategy = "reweighting"
        explanation = "Unequal group influence detected (DPD violation)."

    elif violations["eod"]:
        strategy = "threshold"
        explanation = "Decision boundary bias detected (EOD violation)."

    else:
        strategy = "none"
        explanation = "No actionable bias detected."

    return {
        "recommended_strategy": strategy,
        "severity": severity,
        "explanation": explanation,
    }
