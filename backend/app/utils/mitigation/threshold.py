from fairlearn.postprocessing import ThresholdOptimizer


def apply_threshold_optimizer(
    model,
    X_train,
    y_train,
    sensitive_train,
    grid_size=200,
):
    """
    Wraps the base model using Fairlearn ThresholdOptimizer.
    """

    if isinstance(model, ThresholdOptimizer):
        base_estimator = getattr(model, "estimator_", getattr(model, "estimator", model))
    else:
        base_estimator = model

    predict_method = "predict_proba" if hasattr(base_estimator, "predict_proba") else "decision_function"

    optimizer = ThresholdOptimizer(
    estimator=base_estimator,
    constraints="equalized_odds",
    predict_method=predict_method,
    grid_size=grid_size,
)

    optimizer.fit(
        X_train,
        y_train,
        sensitive_features=sensitive_train,
    )

    return optimizer