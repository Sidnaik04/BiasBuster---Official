def compute_strategy_score(before, after):

    fairness_before = before["fairness"]
    fairness_after = after["fairness"]

    perf_before = before["performance"]
    perf_after = after["performance"]

    # Fairness improvements
    dpd_improve = fairness_before["dpd"] - fairness_after["dpd"]
    eod_improve = fairness_before["eod"] - fairness_after["eod"]
    dir_improve = fairness_after["dir"] - fairness_before["dir"]

    fairness_score = dpd_improve + eod_improve + dir_improve

    # Accuracy penalty
    accuracy_drop = perf_before["accuracy"] - perf_after["accuracy"]

    final_score = fairness_score - accuracy_drop

    return final_score


def rank_strategies(results):

    ranking = []

    for strategy, data in results.items():

        score = compute_strategy_score(data["before"], data["after"])

        ranking.append({"strategy": strategy, "score": score})

    ranking.sort(key=lambda x: x["score"], reverse=True)

    return ranking


def find_best_strategy(strategy_results):

    ranking = rank_strategies(strategy_results)

    best = ranking[0]

    return {"best_strategy": best["strategy"], "ranking": ranking}
