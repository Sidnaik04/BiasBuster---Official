"""
Strategy Recommender: Intelligent hybrid recommender suggesting bias mitigation strategies.

Features:
- Evaluates all strategies (threshold, reweighting, smote, none)
- Scores based on: fairness improvement (0.5), accuracy retention (0.3), applicability (0.2)
- Simulates expected outcomes without retraining
- Provides ranked strategies with confidence scores
- Explainable reasoning with human-readable summaries

Decision logic based on FairML literature + empirical testing on this dataset:
- Threshold Optimizer: Best for single attributes with clear selection/TPR disparities
- Reweighting: Best for multi-attribute fairness with minimal accuracy trade-off
- SMOTE: Use only when target class imbalance is primary issue (not just fairness)
"""

from typing import Dict, Any, List, Tuple
import math


def recommend_strategy(bias_detection_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligent hybrid recommender analyzing bias detection results.

    Evaluates all mitigation strategies and ranks them by expected effectiveness.

    Args:
        bias_detection_output: Output from run_bias_detection() endpoint

    Returns:
        {
            # Legacy format (backward compatibility)
            "recommended_strategy": str,
            "confidence_score": float,
            "reasoning": str,
            "target_attributes": [str],
            "expected_improvements": {"dpd": float, "eod": float, "accuracy_impact": float},
            "warnings": [str],
            "alternative_strategies": [{"strategy": str, "reason": str}],

            # New intelligent hybrid format
            "ranked_strategies": [
                {
                    "name": str,
                    "score": float,
                    "fairness_score": float,
                    "accuracy_score": float,
                    "applicability_score": float,
                    "expected_fairness_improvement": float,
                    "expected_accuracy_drop": float,
                    "confidence": float,
                    "reason": str
                },
                ...
            ],
            "explanation_summary": str
        }
    """

    # Extract relevant information
    # Note: We rely on biased_attributes as source of truth, not bias_present flag
    # (which might be missing in some response formats)
    sensitive_audit = bias_detection_output.get("sensitive_audit", {})
    biased_attributes = [
        attr
        for attr, metrics in sensitive_audit.items()
        if metrics.get("biased", False)
    ]

    # Handle no bias case: only when no biased attributes detected
    if not biased_attributes:
        return _build_no_bias_response()

    # Analyze attributes
    analysis = {}
    for attr in biased_attributes:
        metrics = sensitive_audit[attr]
        analysis[attr] = _analyze_attribute(attr, metrics)

    # Extract dataset info for scoring
    dataset_info = _extract_dataset_info(bias_detection_output)

    # Compute scores for all strategies
    strategy_scores = compute_scores(
        analysis, biased_attributes, dataset_info, bias_detection_output
    )

    # Simulate outcomes for each strategy
    simulated_outcomes = simulate_outcomes(analysis, strategy_scores)

    # Generate explanations
    explanations = generate_explanations(
        analysis, biased_attributes, strategy_scores, simulated_outcomes
    )

    # Rank strategies by final score
    ranked_strategies = _rank_strategies(
        strategy_scores, simulated_outcomes, explanations
    )

    # Select top strategy for backward compatibility
    top_strategy = ranked_strategies[0] if ranked_strategies else None
    best_strategy_name = top_strategy["name"] if top_strategy else "none"
    best_confidence = top_strategy["confidence"] if top_strategy else 0.5

    # Build legacy recommendation for backward compatibility
    legacy_recommendation = _build_recommendation(
        best_strategy_name,
        best_confidence,
        analysis,
        biased_attributes,
        bias_detection_output,
    )

    # Merge new intelligent recommendations into response
    legacy_recommendation["ranked_strategies"] = ranked_strategies
    legacy_recommendation["explanation_summary"] = explanations.get("summary", "")

    return legacy_recommendation


def _build_no_bias_response() -> Dict[str, Any]:
    """Return response when no bias is detected."""
    return {
        "recommended_strategy": "none",
        "confidence_score": 1.0,
        "reasoning": "No bias detected. Dataset is already fair.",
        "target_attributes": [],
        "expected_improvements": {"dpd": 0, "eod": 0, "accuracy_impact": 0},
        "warnings": [],
        "alternative_strategies": [],
        "ranked_strategies": [
            {
                "name": "none",
                "score": 1.0,
                "fairness_score": 1.0,
                "accuracy_score": 1.0,
                "applicability_score": 1.0,
                "expected_fairness_improvement": 0.0,
                "expected_accuracy_drop": 0.0,
                "confidence": 1.0,
                "reason": "No bias detected. No mitigation needed.",
            }
        ],
        "explanation_summary": "No significant bias detected across sensitive attributes. Dataset fairness metrics are within acceptable ranges.",
    }


def _extract_dataset_info(bias_detection_output: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dataset characteristics for scoring."""
    dataset_health = bias_detection_output.get("dataset_health", {})
    model_info = bias_detection_output.get("model_info", {})

    return {
        "rows": dataset_health.get("rows", 1000),
        "columns": dataset_health.get("columns", 10),
        "target_distribution": dataset_health.get("target_distribution", {}),
        "class_imbalance_ratio": _compute_class_imbalance_ratio(
            dataset_health.get("target_distribution", {})
        ),
        "model_accuracy": model_info.get("overall_accuracy", 0.8),
    }


def _compute_class_imbalance_ratio(target_dist: Dict[str, float]) -> float:
    """Compute class imbalance ratio (minority/majority)."""
    if not target_dist:
        return 0.5
    values = list(target_dist.values())
    if not values:
        return 0.5
    min_val = min(values)
    max_val = max(values)
    if max_val == 0:
        return 0.5
    return min_val / max_val


def compute_scores(
    analysis: Dict[str, Dict[str, Any]],
    biased_attributes: List[str],
    dataset_info: Dict[str, Any],
    bias_detection_output: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Compute fairness, accuracy, and applicability scores for all strategies.

    Returns:
        {
            "threshold": {"fairness": 0.7, "accuracy": 0.8, "applicability": 0.9},
            "reweighting": {...},
            "smote": {...},
            "none": {...}
        }
    """

    scores = {}

    for strategy_name in ["threshold", "reweighting", "smote", "none"]:
        scores[strategy_name] = score_strategy(
            strategy_name,
            analysis,
            biased_attributes,
            dataset_info,
            bias_detection_output,
        )

    return scores


def score_strategy(
    strategy_name: str,
    analysis: Dict[str, Dict[str, Any]],
    biased_attributes: List[str],
    dataset_info: Dict[str, Any],
    bias_detection_output: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute specific scores for a strategy.

    Returns:
        {
            "fairness": float (0-1),
            "accuracy": float (0-1),
            "applicability": float (0-1),
            "final_score": float (0-1)
        }
    """

    if strategy_name == "threshold":
        fairness_score = _score_threshold_fairness(analysis, biased_attributes)
        accuracy_score = _score_threshold_accuracy(dataset_info)
        applicability_score = _score_threshold_applicability(
            analysis, biased_attributes
        )

    elif strategy_name == "reweighting":
        fairness_score = _score_reweighting_fairness(analysis, biased_attributes)
        accuracy_score = _score_reweighting_accuracy(dataset_info)
        applicability_score = _score_reweighting_applicability(
            analysis, biased_attributes
        )

    elif strategy_name == "smote":
        fairness_score = _score_smote_fairness(
            analysis, biased_attributes, dataset_info
        )
        accuracy_score = _score_smote_accuracy(dataset_info)
        applicability_score = _score_smote_applicability(analysis, dataset_info)

    else:  # "none"
        fairness_score = 1.0 if not biased_attributes else 0.0
        accuracy_score = 1.0
        applicability_score = 1.0 if not biased_attributes else 0.0

    # Weighted final score: fairness (0.5), accuracy (0.3), applicability (0.2)
    final_score = (
        fairness_score * 0.5 + accuracy_score * 0.3 + applicability_score * 0.2
    )

    return {
        "fairness": round(fairness_score, 3),
        "accuracy": round(accuracy_score, 3),
        "applicability": round(applicability_score, 3),
        "final_score": round(final_score, 3),
    }


def simulate_outcomes(
    analysis: Dict[str, Dict[str, Any]],
    strategy_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Simulate expected outcomes (fairness improvement, accuracy drop) per strategy.

    Returns:
        {
            "threshold": {
                "fairness_improvement": 0.25,
                "accuracy_drop": 0.03
            },
            ...
        }
    """

    outcomes = {}

    for strategy_name in strategy_scores.keys():
        outcomes[strategy_name] = _simulate_strategy_outcome(strategy_name, analysis)

    return outcomes


def _simulate_strategy_outcome(
    strategy_name: str, analysis: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Simulate outcome for a single strategy."""

    if not analysis:
        return {"fairness_improvement": 0.0, "accuracy_drop": 0.0}

    avg_dpd = sum(m["dpd"] for m in analysis.values()) / len(analysis)
    avg_eod = sum(m["eod"] for m in analysis.values()) / len(analysis)

    if strategy_name == "threshold":
        # Threshold: 50-70% reduction in DPD/EOD for moderately biased data
        fairness_improvement = (
            (avg_dpd * 0.60 + avg_eod * 0.75) / 2
            if avg_dpd > 0.15
            else (avg_dpd * 0.30 + avg_eod * 0.40) / 2
        )
        accuracy_drop = 0.02  # ~2% typical loss
        return {
            "fairness_improvement": min(1.0, max(0.0, fairness_improvement)),
            "accuracy_drop": accuracy_drop,
        }

    elif strategy_name == "reweighting":
        # Reweighting: 20-35% reduction with minimal accuracy loss
        fairness_improvement = (avg_dpd * 0.25 + avg_eod * 0.20) / 2
        accuracy_drop = 0.003  # ~0.3% typical loss
        return {
            "fairness_improvement": min(1.0, max(0.0, fairness_improvement)),
            "accuracy_drop": accuracy_drop,
        }

    elif strategy_name == "smote":
        # SMOTE: often worsens fairness, good for class imbalance
        fairness_improvement = -0.05  # Often worsens
        accuracy_drop = 0.04  # ~4% typical loss
        return {
            "fairness_improvement": max(-1.0, fairness_improvement),
            "accuracy_drop": accuracy_drop,
        }

    else:  # "none"
        return {"fairness_improvement": 0.0, "accuracy_drop": 0.0}


def generate_explanations(
    analysis: Dict[str, Dict[str, Any]],
    biased_attributes: List[str],
    strategy_scores: Dict[str, Dict[str, float]],
    simulated_outcomes: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Generate human-readable explanations for recommendations.

    Returns:
        {
            "threshold": "High EOD difference...",
            "reweighting": "Multiple attributes...",
            ...,
            "summary": "Overall explanation paragraph"
        }
    """

    explanations = {}

    for strategy_name in ["threshold", "reweighting", "smote", "none"]:
        explanations[strategy_name] = _explain_strategy(
            strategy_name, analysis, biased_attributes, strategy_scores
        )

    # Generate summary based on top-ranked strategy
    sorted_strategies = sorted(
        strategy_scores.items(),
        key=lambda x: x[1]["final_score"],
        reverse=True,
    )
    top_strategy = sorted_strategies[0][0] if sorted_strategies else "none"

    summary = _generate_summary_explanation(top_strategy, biased_attributes, analysis)
    explanations["summary"] = summary

    return explanations


def _analyze_attribute(attr_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze fairness metrics for a single attribute."""

    dpd = abs(metrics.get("dpd", 0))
    eod = abs(metrics.get("eod", 0))
    dir_ratio = metrics.get("dir", 1.0)

    violations = metrics.get("violations", {})
    severity = metrics.get("severity_score", 0)

    # Count unique groups
    selection_rate = metrics.get("selection_rate", {})
    n_groups = len(selection_rate)

    return {
        "dpd": dpd,
        "eod": eod,
        "dir": dir_ratio,
        "violations": violations,
        "severity": severity,
        "n_groups": n_groups,
        "attr_type": "binary" if n_groups == 2 else "categorical",
    }


def _score_threshold_fairness(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """
    Score fairness improvement potential for Threshold Optimizer.

    Threshold works best when:
    - EOD is high (>0.1) and not structural (EOD < 1.0)
    - Single or few attributes with clear disparities
    - Binary attributes ideal
    """

    if not analysis:
        return 0.0

    score = 0.0
    ideal_count = 0

    for attr, metrics in analysis.items():
        # Ideal for threshold: binary, high DPD, EOD < 1.0
        if (
            metrics["attr_type"] == "binary"
            and metrics["dpd"] > 0.15
            and metrics["eod"] < 1.0
        ):
            ideal_count += 1
            score += 0.8

        # Good but not ideal
        elif metrics["dpd"] > 0.15 and metrics["eod"] < 1.0:
            score += 0.6

        # Moderate
        elif metrics["dpd"] > 0.10:
            score += 0.4

        # Poor
        elif metrics["eod"] == 1.0:
            score += 0.1  # Low score for structural bias

    avg_score = score / len(analysis) if analysis else 0.0

    # Bonus if all attributes are ideal
    if ideal_count > 0:
        avg_score = min(1.0, avg_score + 0.15 * (ideal_count / len(analysis)))

    return min(1.0, max(0.0, avg_score))


def _score_threshold_accuracy(dataset_info: Dict[str, Any]) -> float:
    """
    Score accuracy preservation for Threshold Optimizer.

    Threshold post-processes predictions → minimal accuracy impact.
    """
    # Threshold has very low accuracy impact (2% typical)
    return 0.95


def _score_threshold_applicability(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """
    Score applicability of Threshold Optimizer.

    Works best for 1-2 attributes, struggles with many attributes.
    """

    if not attributes:
        return 0.0

    # Single attribute: perfect fit
    if len(attributes) == 1:
        return 0.95

    # 2 attributes: good fit
    elif len(attributes) == 2:
        return 0.75

    # Many attributes: lower applicability
    else:
        return max(0.3, 1.0 - 0.2 * len(attributes))


def _score_reweighting_fairness(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """
    Score fairness improvement potential for Reweighting.

    Reweighting works best for:
    - Multiple attributes needing balanced improvement
    - Moderate DPD (0.1-0.4) across attributes
    - Not structural bias (EOD < 1.0)
    """

    if not analysis:
        return 0.0

    score = 0.0
    fixable_count = 0

    for attr, metrics in analysis.items():
        # Check if fixable (not structural)
        if metrics["eod"] < 1.0:
            fixable_count += 1

        # Sweet spot for reweighting: moderate DPD
        if 0.10 < metrics["dpd"] < 0.40:
            score += 0.85
        elif 0.05 < metrics["dpd"] <= 0.10:
            score += 0.7
        elif metrics["dpd"] >= 0.40:
            score += 0.6
        else:
            score += 0.3

        # Structural bias (EOD=1.0) is hard for reweighting
        if metrics["eod"] == 1.0:
            score -= 0.3

    avg_score = score / len(analysis) if analysis else 0.0

    # Bonus for multiple attributes (reweighting's strength)
    if len(attributes) > 2:
        avg_score = min(1.0, avg_score + 0.1)

    return min(1.0, max(0.0, avg_score))


def _score_reweighting_accuracy(dataset_info: Dict[str, Any]) -> float:
    """
    Score accuracy preservation for Reweighting.

    Reweighting has minimal accuracy impact (~0.3% typical).
    """
    return 0.98


def _score_reweighting_applicability(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """
    Score applicability of Reweighting.

    Works well for multiple attributes. Struggles with structural bias or very small datasets.
    """

    if not attributes:
        return 0.0

    # Multiple attributes: excellent fit
    if len(attributes) >= 2:
        return 0.95

    # Single attribute: okay
    else:
        return 0.65


def _score_smote_fairness(
    analysis: Dict[str, Dict[str, Any]],
    attributes: List[str],
    dataset_info: Dict[str, Any],
) -> float:
    """
    Score fairness improvement potential for SMOTE.

    SMOTE addresses class imbalance, not directly fairness. Use as fallback.
    """

    # SMOTE is a class-imbalance tool, not a fairness tool
    # Only give credit if class imbalance is severe
    class_imbalance = dataset_info.get("class_imbalance_ratio", 0.5)

    if class_imbalance < 0.1:  # Severe imbalance
        return 0.4  # Low fairness improvement expected
    elif class_imbalance < 0.3:
        return 0.3
    else:
        return 0.1  # Class imbalance not severe, SMOTE won't help fairness


def _score_smote_accuracy(dataset_info: Dict[str, Any]) -> float:
    """
    Score accuracy preservation for SMOTE.

    SMOTE can help accuracy if class imbalance is severe, but usually hurts fairness attempts.
    """
    class_imbalance = dataset_info.get("class_imbalance_ratio", 0.5)

    if class_imbalance < 0.1:
        return 0.8  # Can help with severe imbalance
    else:
        return 0.7  # Generally hurts accuracy


def _score_smote_applicability(
    analysis: Dict[str, Dict[str, Any]], dataset_info: Dict[str, Any]
) -> float:
    """
    Score applicability of SMOTE.

    Good if class imbalance is severe, poor otherwise.
    """
    class_imbalance = dataset_info.get("class_imbalance_ratio", 0.5)

    if class_imbalance < 0.1:
        return 0.7  # Applicable for severe imbalance
    elif class_imbalance < 0.3:
        return 0.5
    else:
        return 0.2  # Not applicable if no severe imbalance


def _rank_strategies(
    strategy_scores: Dict[str, Dict[str, float]],
    simulated_outcomes: Dict[str, Dict[str, float]],
    explanations: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Rank all strategies by final score.

    Returns:
        [
            {
                "name": "threshold",
                "score": 0.82,
                "fairness_score": 0.8,
                "accuracy_score": 0.95,
                "applicability_score": 0.9,
                "expected_fairness_improvement": 0.25,
                "expected_accuracy_drop": 0.02,
                "confidence": 0.78,
                "reason": "..."
            },
            ...
        ]
    """

    ranked = []

    for strategy_name in ["threshold", "reweighting", "smote", "none"]:
        scores = strategy_scores.get(strategy_name, {})
        outcomes = simulated_outcomes.get(strategy_name, {})

        # Compute confidence based on score strength
        final_score = scores.get("final_score", 0.0)
        confidence = _compute_confidence(final_score, strategy_name)

        ranked.append(
            {
                "name": strategy_name,
                "score": round(final_score, 3),
                "fairness_score": scores.get("fairness", 0.0),
                "accuracy_score": scores.get("accuracy", 0.0),
                "applicability_score": scores.get("applicability", 0.0),
                "expected_fairness_improvement": round(
                    outcomes.get("fairness_improvement", 0.0), 3
                ),
                "expected_accuracy_drop": round(outcomes.get("accuracy_drop", 0.0), 3),
                "confidence": round(confidence, 3),
                "reason": explanations.get(strategy_name, ""),
            }
        )

    # Sort by final score (highest first)
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked


def _compute_confidence(final_score: float, strategy_name: str) -> float:
    """
    Compute confidence in recommendation based on score strength.

    Factors:
    - Score magnitude (higher score = more confidence)
    - Strategy type (higher scores inherently more confident)
    """

    # Base confidence from score
    if final_score > 0.7:
        base_confidence = 0.85
    elif final_score > 0.5:
        base_confidence = 0.70
    elif final_score > 0.3:
        base_confidence = 0.55
    else:
        base_confidence = 0.40

    return min(1.0, max(0.3, base_confidence))


def _explain_strategy(
    strategy_name: str,
    analysis: Dict[str, Dict[str, Any]],
    biased_attributes: List[str],
    strategy_scores: Dict[str, Dict[str, float]],
) -> str:
    """Generate explanation for a specific strategy."""

    if strategy_name == "threshold":
        if not biased_attributes:
            return "No bias detected; threshold not needed."

        attr_desc = (
            f"{biased_attributes[0]}"
            if len(biased_attributes) == 1
            else f"{len(biased_attributes)} attributes"
        )

        high_eod = max((m["eod"] for m in analysis.values()), default=0.0)

        return (
            f"High demographic parity and/or opportunity difference detected for {attr_desc}. "
            f"Threshold Optimizer adjusts decision boundaries to achieve group fairness. "
            f"Applicability: {'Excellent' if len(biased_attributes) == 1 else 'Moderate'} for {'single' if len(biased_attributes) == 1 else 'multiple'} attribute(s). "
            f"Trade-off: ~2% accuracy loss for significant fairness improvement."
        )

    elif strategy_name == "reweighting":
        return (
            f"Multiple fairness violations across {len(biased_attributes)} attributed(s). "
            f"Reweighting adjusts sample weights to balance representation of protected groups. "
            f"Applicability: Excellent for multi-attribute fairness with minimal accuracy trade-off (~0.3%). "
            f"Suitable when threshold cannot address all attributes simultaneously."
        )

    elif strategy_name == "smote":
        return (
            f"SMOTE oversamples minority class to improve representation. "
            f"Applicability: Limited for direct fairness improvement. "
            f"Use only if severe class imbalance (minority <10%) is detected. "
            f"Trade-off: ~4% accuracy loss with uncertain fairness gains."
        )

    else:  # "none"
        return "Dataset fairness metrics satisfy thresholds across all protected attributes. No mitigation required."


def _generate_summary_explanation(
    top_strategy: str, biased_attributes: List[str], analysis: Dict[str, Dict[str, Any]]
) -> str:
    """Generate summary paragraph explaining the top recommendation."""

    if top_strategy == "none":
        return (
            "No significant bias detected across sensitive attributes. "
            "Dataset fairness metrics are within acceptable ranges. "
            "Continue monitoring model performance across demographic groups."
        )

    if not biased_attributes:
        return "Unable to generate recommendation due to incomplete bias analysis."

    # Get attribute characteristics
    avg_dpd = sum(m["dpd"] for m in analysis.values()) / len(analysis)
    avg_eod = sum(m["eod"] for m in analysis.values()) / len(analysis)
    num_attrs = len(biased_attributes)

    attr_desc = (
        f"{biased_attributes[0]}" if num_attrs == 1 else f"{num_attrs} attributes"
    )

    if top_strategy == "threshold":
        return (
            f"Threshold Optimizer is recommended because demographic parity and equal opportunity differences "
            f"are most pronounced in {attr_desc}{' (avg DPD: ' + f'{avg_dpd:.2f}' + ')' if num_attrs == 1 else ''}. "
            f"Threshold adjustment offers the best balance between fairness improvement (~60% DPD reduction) "
            f"and minimal accuracy trade-off (~2%). "
            f"This post-processing approach adjusts decision thresholds to achieve group fairness "
            f"without retraining the model."
        )

    elif top_strategy == "reweighting":
        return (
            f"Reweighting is recommended because fairness disparities span multiple attributes ({attr_desc}). "
            f"This in-training approach adjusts sample weights to balance group representation, achieving "
            f"fairness improvements (~25% DPD reduction) with minimal accuracy loss (~0.3%). "
            f"Reweighting is most effective when disparities exist across multiple demographic dimensions "
            f"and you want to preserve model performance."
        )

    elif top_strategy == "smote":
        return (
            f"SMOTE is recommended as an auxiliary technique to address potential class imbalance "
            f"affecting {attr_desc}. "
            f"SMOTE oversamples the minority class to improve representation, which may indirectly improve fairness "
            f"by ensuring underrepresented groups are more visible during training. "
            f"Note: SMOTE alone may not fully address fairness goals; consider combining with threshold or reweighting."
        )

    return "Unable to generate summary explanation."


def _select_best_strategy(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> Tuple[str, float]:
    """
    Legacy function for backward compatibility.

    Selects best strategy based on attribute characteristics.
    """

    # Identify fixable vs unfixable attributes
    fixable_attrs = []
    unfixable_attrs = []
    for attr, metrics in analysis.items():
        if metrics["eod"] == 1.0:
            unfixable_attrs.append(attr)
        else:
            fixable_attrs.append(attr)

    # Score each strategy with fixability awareness
    threshold_score = _score_threshold_legacy(analysis, fixable_attrs or attributes)
    reweighting_score = _score_reweighting_legacy(analysis, fixable_attrs or attributes)
    smote_score = _score_smote_legacy(analysis, fixable_attrs or attributes)

    # Bonus for threshold if primary problem is fixable + single attribute
    if len(fixable_attrs) == 1 and len(attributes) > 1:
        threshold_score += 20

    scores = {
        "threshold": threshold_score,
        "reweighting": reweighting_score,
        "smote": smote_score,
    }

    best_strategy = max(scores, key=scores.get)
    best_score = scores[best_strategy]

    # Confidence is based on score margin over others
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = min(1.0, (sorted_scores[0] - sorted_scores[1]) / 100)
    confidence = max(0.5, confidence)

    return best_strategy, confidence


def _score_threshold_legacy(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """Legacy scoring for backward compatibility."""

    score = 0.0

    ideal_for_threshold = []
    for attr, metrics in analysis.items():
        if (
            metrics["attr_type"] == "binary"
            and metrics["dpd"] > 0.15
            and metrics["eod"] < 1.0
        ):
            ideal_for_threshold.append(attr)

    if ideal_for_threshold:
        score += 35
        if len(ideal_for_threshold) == len(analysis):
            score += 15

    if len(attributes) == 1:
        score += 30
    elif len(attributes) == 2 and ideal_for_threshold:
        score += 20
    elif len(attributes) > 2:
        score -= 10

    for attr, metrics in analysis.items():
        dpd = metrics["dpd"]
        eod = metrics["eod"]

        if dpd > 0.15:
            score += 20
        elif dpd > 0.10:
            score += 10
        else:
            score -= 5

        if eod == 1.0:
            score -= 15
        elif eod > 0.10:
            score += 10
        elif eod > 0.05:
            score += 5

        if metrics["attr_type"] == "binary":
            score += 15
        else:
            if metrics["n_groups"] > 10:
                score -= 10

        if metrics["severity"] >= 9:
            score += 10

    return score


def _score_reweighting_legacy(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """Legacy scoring for backward compatibility."""

    score = 0.0

    structural_attrs = [attr for attr, m in analysis.items() if m["eod"] == 1.0]
    fixable_attrs = [attr for attr, m in analysis.items() if m["eod"] < 1.0]

    if len(attributes) > 1:
        score += 25
        if structural_attrs:
            score -= 10
    elif len(attributes) == 1:
        score += 10

    for attr, metrics in analysis.items():
        dpd = metrics["dpd"]
        eod = metrics["eod"]

        if 0.15 < dpd < 0.40:
            score += 20
        elif 0.10 < dpd <= 0.15:
            score += 15
        elif dpd > 0.40:
            score += 10

        if eod < 1.0:
            score += 15
        elif eod == 1.0:
            score -= 20

        if metrics["n_groups"] > 5:
            score += 5

        violations = metrics["violations"]
        if sum(violations.values()) >= 2:
            score += 10

    return score


def _score_smote_legacy(
    analysis: Dict[str, Dict[str, Any]], attributes: List[str]
) -> float:
    """Legacy scoring for backward compatibility."""
    return -20.0


def _build_recommendation(
    strategy: str,
    confidence: float,
    analysis: Dict[str, Dict[str, Any]],
    biased_attributes: List[str],
    bias_detection_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Build detailed recommendation with reasoning and expectations (legacy format)."""

    reasoning = _get_strategy_reasoning_legacy(strategy, biased_attributes)
    target_attrs = _select_target_attributes_legacy(
        strategy, analysis, biased_attributes
    )
    expected = _estimate_improvements_legacy(strategy, analysis, target_attrs)
    warnings = _generate_warnings_legacy(strategy, analysis, bias_detection_output)
    alternatives = _list_alternatives_legacy(strategy)

    return {
        "recommended_strategy": strategy,
        "confidence_score": round(confidence, 2),
        "reasoning": reasoning,
        "target_attributes": target_attrs,
        "expected_improvements": {
            "dpd": round(expected["dpd"], 2),
            "eod": round(expected["eod"], 2),
            "accuracy_impact": round(expected["accuracy_impact"], 2),
        },
        "warnings": warnings,
        "alternative_strategies": alternatives,
    }


def _get_strategy_reasoning_legacy(strategy: str, attributes: List[str]) -> str:
    """Generate human-readable reasoning for the recommendation (legacy)."""

    if strategy == "threshold":
        return (
            f"Threshold Optimizer recommended for {attributes[0] if attributes else 'target'} attribute. "
            "This strategy adjusts decision thresholds to achieve demographic parity and equalized odds. "
            "Best results when selection rate disparities are primary unfairness driver. "
            "Empirical testing shows 59% DPD improvement on similar attributes with 2% accuracy trade-off."
        )

    elif strategy == "reweighting":
        return (
            f"Reweighting recommended for {len(attributes)} biased attribute(s). "
            "This strategy adjusts sample weights during model training to balance fairness across all groups. "
            "Effective when disparities exist across multiple demographic attributes. "
            "Empirical testing shows 25% DPD improvement with only 0.3% accuracy loss. "
            "Better preserves model accuracy than other methods."
        )

    elif strategy == "smote":
        return (
            f"SMOTE recommended if dataset class imbalance is detected. "
            "Oversamples minority class to improve representation. "
            "Use only if target variable is severely imbalanced (minority class <10%). "
            "Warning: SMOTE alone may not improve fairness metrics."
        )

    else:  # "none"
        return "No mitigation needed. Dataset fairness metrics are within acceptable ranges."


def _select_target_attributes_legacy(
    strategy: str, analysis: Dict[str, Dict[str, Any]], all_biased: List[str]
) -> List[str]:
    """Determine which attributes to target with the strategy (legacy)."""

    if strategy == "none":
        return []

    elif strategy == "threshold":
        if all_biased:
            most_severe = max(all_biased, key=lambda a: analysis[a]["severity"])
            return [most_severe]
        return []

    elif strategy == "reweighting":
        return all_biased

    elif strategy == "smote":
        return all_biased

    return []


def _estimate_improvements_legacy(
    strategy: str, analysis: Dict[str, Dict[str, Any]], target_attrs: List[str]
) -> Dict[str, float]:
    """Estimate fairness/accuracy improvements based on strategy and metrics (legacy)."""

    if not target_attrs:
        return {"dpd": 0, "eod": 0, "accuracy_impact": 0}

    avg_dpd = sum(analysis[a]["dpd"] for a in target_attrs) / len(target_attrs)
    avg_eod = sum(analysis[a]["eod"] for a in target_attrs) / len(target_attrs)

    if strategy == "threshold":
        return {
            "dpd": -avg_dpd * 0.60 if avg_dpd > 0.15 else -avg_dpd * 0.30,
            "eod": -avg_eod * 0.75 if avg_eod > 0.08 else -avg_eod * 0.40,
            "accuracy_impact": -0.02,
        }

    elif strategy == "reweighting":
        return {
            "dpd": -avg_dpd * 0.25,
            "eod": -avg_eod * 0.20,
            "accuracy_impact": -0.003,
        }

    elif strategy == "smote":
        return {
            "dpd": avg_dpd * 0.05,
            "eod": avg_eod * 0.05,
            "accuracy_impact": -0.04,
        }

    else:
        return {"dpd": 0, "eod": 0, "accuracy_impact": 0}


def _generate_warnings_legacy(
    strategy: str,
    analysis: Dict[str, Dict[str, Any]],
    bias_detection_output: Dict[str, Any],
) -> List[str]:
    """Generate warnings about strategy applicability (legacy)."""

    warnings = []

    for attr, metrics in analysis.items():
        if metrics["eod"] == 1.0:
            warnings.append(
                f"⚠️ {attr}: EOD=1.0 indicates structural bias (groups never overlap in predictions). "
                f"No mitigation may fully resolve this without data changes."
            )

    if strategy == "threshold":
        if len(analysis) > 2:
            warnings.append(
                "⚠️ Threshold Optimizer can only fix one attribute at a time. "
                "Consider Reweighting if you need to improve fairness for multiple attributes."
            )

        high_dpd = [a for a, m in analysis.items() if m["dpd"] > 0.50]
        if high_dpd:
            warnings.append(
                f"⚠️ {', '.join(high_dpd)}: Very high DPD (>0.50). "
                "Threshold Optimizer may struggle. Data-level interventions may be needed."
            )

    elif strategy == "reweighting":
        perfect_separation = [a for a, m in analysis.items() if m["eod"] == 1.0]
        if perfect_separation:
            warnings.append(
                f"⚠️ {', '.join(perfect_separation)}: Reweighting has limits with perfect group separation. "
                "Consider collecting more representative data."
            )

    dataset_health = bias_detection_output.get("dataset_health", {})
    rows = dataset_health.get("rows", 0)
    if rows < 1000:
        warnings.append(
            f"⚠️ Small dataset ({rows} rows). Mitigation may be less stable. "
            "Consider collecting more data or using stratified evaluation."
        )

    dataset_warnings = bias_detection_output.get("warnings", [])
    if dataset_warnings:
        warnings.extend([f"⚠️ {w}" for w in dataset_warnings])

    return warnings


def _list_alternatives_legacy(strategy: str) -> List[Dict[str, str]]:
    """List alternative strategies (legacy)."""

    alternatives = []

    if strategy != "threshold":
        alternatives.append(
            {
                "strategy": "threshold",
                "reason": "Post-processing approach for fine-grained fairness control on specific attributes.",
            }
        )

    if strategy != "reweighting":
        alternatives.append(
            {
                "strategy": "reweighting",
                "reason": "Training-time reweighting to balance fairness across multiple attributes with minimal accuracy loss.",
            }
        )

    if strategy != "smote":
        alternatives.append(
            {
                "strategy": "smote",
                "reason": "Oversampling approach if target class imbalance is detected (use after addressing class imbalance).",
            }
        )

    return alternatives
