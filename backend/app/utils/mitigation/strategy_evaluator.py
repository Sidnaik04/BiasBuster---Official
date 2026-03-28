"""
Strategy Evaluator: Ranks mitigation strategies by comparing fairness improvements vs. accuracy trade-offs.

Scoring Formula:
    fairness_score = (DPD_improvement + EOD_improvement + DI_improvement)
    accuracy_penalty = (accuracy_before - accuracy_after)
    final_score = fairness_score - 0.5 * accuracy_penalty

The 0.5 weight prioritizes fairness while acknowledging accuracy matters.
Adjust the weight based on your use case (higher = prioritize fairness more).
"""

from typing import Dict, List, Any, Tuple


def compute_strategy_score(
    before_metrics: Dict[str, Any],
    after_metrics: Dict[str, Any],
    fairness_weight: float = 1.0,
    accuracy_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive score for a strategy comparing before/after metrics.

    Args:
        before_metrics: Fairness + performance metrics before mitigation
            Can be either:
            - Flat: {"fairness": {"dpd": float, ...}, "performance": {...}}
            - Nested by attribute: {"gender": {"dpd": float, ..., "performance": {...}}}
        after_metrics: Fairness + performance metrics after mitigation
        fairness_weight: Weight for fairness improvements (default 1.0)
        accuracy_weight: Weight for accuracy penalty (default 0.5)

    Returns:
        {
            "total_score": float,
            "fairness_score": float,
            "accuracy_penalty": float,
            "dpd_improvement": float,
            "eod_improvement": float,
            "di_improvement": float,
            "accuracy_drop": float,
            "precision_drop": float,
            "recall_drop": float,
            "f1_drop": float
        }
    """

    # Handle both flat and attribute-nested metric structures
    if "fairness" in before_metrics:
        # Flat structure
        fairness_before = before_metrics.get("fairness", {})
        perf_before = before_metrics.get("performance", {})
    else:
        # Attribute-nested structure: aggregate across attributes
        fairness_before = {}
        perf_before = {}
        for attr_name, attr_metrics in before_metrics.items():
            if isinstance(attr_metrics, dict):
                # Extract fairness metrics
                if "dpd" not in fairness_before:
                    fairness_before["dpd"] = attr_metrics.get("dpd", 0)
                    fairness_before["eod"] = attr_metrics.get("eod", 0)
                    fairness_before["dir"] = attr_metrics.get("dir", 1.0)
                # Extract performance metrics
                perf = attr_metrics.get("performance", {})
                if perf and not perf_before:
                    perf_before = perf

    if "fairness" in after_metrics:
        # Flat structure
        fairness_after = after_metrics.get("fairness", {})
        perf_after = after_metrics.get("performance", {})
    else:
        # Attribute-nested structure: aggregate across attributes
        fairness_after = {}
        perf_after = {}
        for attr_name, attr_metrics in after_metrics.items():
            if isinstance(attr_metrics, dict):
                # Extract fairness metrics
                if "dpd" not in fairness_after:
                    fairness_after["dpd"] = attr_metrics.get("dpd", 0)
                    fairness_after["eod"] = attr_metrics.get("eod", 0)
                    fairness_after["dir"] = attr_metrics.get("dir", 1.0)
                # Extract performance metrics
                perf = attr_metrics.get("performance", {})
                if perf and not perf_after:
                    perf_after = perf

    # Fairness improvements (higher is better)
    dpd_improve = abs(fairness_before.get("dpd", 0)) - abs(fairness_after.get("dpd", 0))
    eod_improve = abs(fairness_before.get("eod", 0)) - abs(fairness_after.get("eod", 0))
    # DI: higher is better (closer to 1.0), so inverse calculation
    dir_improve = fairness_after.get("dir", 1.0) - fairness_before.get("dir", 1.0)

    fairness_score = dpd_improve + eod_improve + dir_improve

    # Performance penalties (lower is better, so subtract)
    accuracy_drop = perf_before.get("accuracy", 0) - perf_after.get("accuracy", 0)
    precision_drop = perf_before.get("precision", 0) - perf_after.get("precision", 0)
    recall_drop = perf_before.get("recall", 0) - perf_after.get("recall", 0)
    f1_drop = perf_before.get("f1", 0) - perf_after.get("f1", 0)

    # Combined score: fairness improvements minus accuracy penalty
    # Using weighted formula to balance fairness vs accuracy
    total_score = (fairness_weight * fairness_score) - (accuracy_weight * accuracy_drop)

    return {
        "total_score": total_score,
        "fairness_score": fairness_score,
        "accuracy_penalty": accuracy_drop,
        "dpd_improvement": dpd_improve,
        "eod_improvement": eod_improve,
        "di_improvement": dir_improve,
        "accuracy_drop": accuracy_drop,
        "precision_drop": precision_drop,
        "recall_drop": recall_drop,
        "f1_drop": f1_drop,
    }


def rank_strategies(
    strategy_results: Dict[str, Dict[str, Any]],
    fairness_weight: float = 1.0,
    accuracy_weight: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Rank all strategies by their effectiveness.

    Args:
        strategy_results: Results from all strategies in format:
            {
                "reweighting": {
                    "before_metrics": {...},
                    "after_metrics": {...}
                },
                "threshold": {...},
                "smote": {...}
            }
        fairness_weight: Weight for fairness (default 1.0)
        accuracy_weight: Weight for accuracy penalty (default 0.5)

    Returns:
        Sorted list of strategies with scores:
            [
                {
                    "rank": 1,
                    "strategy": "threshold",
                    "total_score": 0.85,
                    "fairness_score": 0.90,
                    "accuracy_penalty": -0.02,
                    ...
                },
                ...
            ]
    """

    ranking = []

    for strategy_name, results in strategy_results.items():
        before = results.get("before_metrics", {})
        after = results.get("after_metrics", {})

        # Handle missing data
        if not before or not after:
            continue

        score_data = compute_strategy_score(
            before, after, fairness_weight, accuracy_weight
        )

        ranking.append(
            {
                "strategy": strategy_name,
                **score_data,
            }
        )

    # Sort by total_score descending (higher is better)
    ranking.sort(key=lambda x: x["total_score"], reverse=True)

    # Add rank
    for idx, item in enumerate(ranking, 1):
        item["rank"] = idx

    return ranking


def find_best_strategy(
    strategy_results: Dict[str, Dict[str, Any]],
    fairness_weight: float = 1.0,
    accuracy_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Find the best strategy overall.

    Args:
        strategy_results: Results from all strategies
        fairness_weight: Weight for fairness
        accuracy_weight: Weight for accuracy penalty

    Returns:
        {
            "best_strategy": "threshold",
            "best_score": 0.85,
            "ranking": [
                {"rank": 1, "strategy": "threshold", ...},
                {"rank": 2, "strategy": "reweighting", ...},
                {"rank": 3, "strategy": "smote", ...}
            ]
        }
    """

    ranking = rank_strategies(strategy_results, fairness_weight, accuracy_weight)

    if not ranking:
        return {
            "best_strategy": None,
            "best_score": None,
            "ranking": [],
            "error": "No valid strategy results provided",
        }

    best = ranking[0]

    return {
        "best_strategy": best["strategy"],
        "best_score": round(best["total_score"], 4),
        "ranking": ranking,
    }


def generate_comparison_report(
    strategy_results: Dict[str, Dict[str, Any]],
    fairness_weight: float = 1.0,
    accuracy_weight: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate a detailed comparison report of all strategies.

    Args:
        strategy_results: Results from all strategies
        fairness_weight: Weight for fairness
        accuracy_weight: Weight for accuracy penalty

    Returns:
        Detailed report with insights and recommendations
    """

    best_result = find_best_strategy(strategy_results, fairness_weight, accuracy_weight)
    ranking = best_result["ranking"]
    best_strategy = best_result["best_strategy"]

    if not ranking:
        return {"status": "error", "message": "No valid strategy results to compare"}

    # Generate insights
    insights = _generate_insights(ranking, strategy_results)

    return {
        "status": "success",
        "best_strategy": best_strategy,
        "best_score": best_result["best_score"],
        "ranking": [
            {
                "rank": r["rank"],
                "strategy": r["strategy"],
                "total_score": round(r["total_score"], 4),
                "fairness_improvement": round(r["fairness_score"], 4),
                "accuracy_impact": round(
                    -r["accuracy_penalty"], 4
                ),  # Negative for clarity
                "dpd_improvement": round(r["dpd_improvement"], 4),
                "eod_improvement": round(r["eod_improvement"], 4),
                "di_improvement": round(r["di_improvement"], 4),
            }
            for r in ranking
        ],
        "insights": insights,
        "recommendation": _get_recommendation(best_strategy, ranking),
    }


def _generate_insights(
    ranking: List[Dict[str, Any]], strategy_results: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Generate strategic insights from the ranking."""

    insights = []

    if not ranking:
        return insights

    best = ranking[0]

    # Insight 1: Best strategy clear winner?
    if len(ranking) > 1:
        score_gap = best["total_score"] - ranking[1]["total_score"]
        if score_gap > 0.3:
            insights.append(
                f"🏆 {best['strategy'].upper()} is a clear winner with {score_gap:.3f} "
                f"point lead over {ranking[1]['strategy']}."
            )
        elif score_gap > 0.1:
            insights.append(
                f"🎯 {best['strategy'].upper()} wins, but {ranking[1]['strategy']} is competitive "
                f"(only {score_gap:.3f} behind)."
            )
        else:
            insights.append(
                f"⚖️ {best['strategy'].upper()} and {ranking[1]['strategy']} are very similar. "
                f"Consider other factors like implementation complexity."
            )

    # Insight 2: What's the trade-off?
    fairness_score = best["fairness_score"]
    accuracy_penalty = best["accuracy_penalty"]

    if fairness_score > 0.2 and accuracy_penalty > 0.02:
        insights.append(
            f"📊 {best['strategy'].upper()} sacrifices {accuracy_penalty:.1%} accuracy for "
            f"{fairness_score:.3f} fairness improvement. Acceptable trade-off?"
        )
    elif fairness_score > 0.2 and accuracy_penalty < 0.01:
        insights.append(
            f"✨ {best['strategy'].upper()} achieves {fairness_score:.3f} fairness improvement "
            f"with minimal accuracy loss ({accuracy_penalty:.1%})."
        )
    elif fairness_score < 0.1:
        insights.append(
            f"⚠️ {best['strategy'].upper()} achieves only {fairness_score:.3f} fairness improvement. "
            f"Bias may be structural and hard to fix."
        )

    # Insight 3: Consistency across fairness metrics
    dpd = best["dpd_improvement"]
    eod = best["eod_improvement"]
    di = best["di_improvement"]

    improvements = [dpd, eod, di]
    balanced = sum(1 for x in improvements if x > 0.05)

    if balanced == 3:
        insights.append(
            f"🎪 {best['strategy'].upper()} improves all fairness metrics (DPD, EOD, DI) equally."
        )
    elif balanced == 2:
        insights.append(
            f"📈 {best['strategy'].upper()} improves 2 of 3 fairness metrics. Some attributes "
            f"may need additional attention."
        )
    elif balanced == 1:
        insights.append(
            f"🔍 {best['strategy'].upper()} primarily improves one fairness metric. "
            f"Other disparities may persist."
        )
    else:
        insights.append(
            f"⚠️ {best['strategy'].upper()} shows limited fairness improvements. "
            f"Consider data-level interventions."
        )

    return insights


def _get_recommendation(best_strategy: str, ranking: List[Dict[str, Any]]) -> str:
    """Generate actionable recommendation."""

    if not best_strategy:
        return "Unable to recommend. Please check input data."

    recommendations = {
        "threshold": (
            f"✅ Deploy Threshold Optimizer. It achieved the best fairness-accuracy balance. "
            f"Apply threshold optimization to the primary biased attribute for targeted fairness improvements."
        ),
        "reweighting": (
            f"✅ Deploy Reweighting. It balances fairness across multiple attributes with minimal accuracy loss. "
            f"Use during model retraining to assign higher weights to underrepresented groups."
        ),
        "smote": (
            f"⚠️ SMOTE ranked best, but review results carefully. "
            f"SMOTE works best when class imbalance is the primary issue. "
            f"Consider threshold or reweighting if fairness improvements are modest."
        ),
    }

    return recommendations.get(
        best_strategy, f"Deploy {best_strategy.upper()} strategy."
    )
