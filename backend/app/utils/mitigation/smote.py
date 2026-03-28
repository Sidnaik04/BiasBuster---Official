from imblearn.over_sampling import SMOTE
from sklearn.base import clone


def apply_smote(X, y, sensitive_features, model):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, SMOTENC
    from sklearn.base import clone
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np

    if isinstance(model, ThresholdOptimizer):
        base_pipe = model.estimator
    else:
        base_pipe = model

    base_pipe = clone(base_pipe)

    # Generate unique SMOTE step name
    smote_counter = 0
    if hasattr(base_pipe, "steps") or isinstance(base_pipe, Pipeline):
        # Count existing SMOTE steps to generate unique name
        for step_name, _ in base_pipe.steps:
            if step_name.startswith("smote"):
                try:
                    counter = int(step_name.split("_")[-1])
                    smote_counter = max(smote_counter, counter + 1)
                except (ValueError, IndexError):
                    smote_counter += 1

    smote_name = (
        f"smote_{smote_counter}"
        if smote_counter > 0
        or (hasattr(base_pipe, "steps") or isinstance(base_pipe, Pipeline))
        else "smote"
    )

    if hasattr(base_pipe, "steps") or isinstance(base_pipe, Pipeline):
        steps = base_pipe.steps
        sampler = SMOTE(random_state=42)
        new_steps = steps[:-1] + [(smote_name, sampler), steps[-1]]
    else:
        cat_indices = []
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
            cat_indices = [X.columns.get_loc(c) for c in cat_cols]

        if len(cat_indices) > 0:
            sampler = SMOTENC(categorical_features=cat_indices, random_state=42)
        else:
            sampler = SMOTE(random_state=42)

        new_steps = [(smote_name, sampler), ("model", base_pipe)]

    new_model_base = ImbPipeline(new_steps)

    if isinstance(model, ThresholdOptimizer):
        new_model = ThresholdOptimizer(
            estimator=new_model_base,
            constraints=model.constraints,
            predict_method=model.predict_method,
            grid_size=getattr(model, "grid_size", 1000),
        )
        new_model.fit(X, y, sensitive_features=sensitive_features)
    else:
        new_model = new_model_base
        new_model.fit(X, y)

    y_counts = (
        np.bincount(y)
        if not isinstance(y, pd.Series)
        else y.value_counts(dropna=True).values
    )
    rows_after = int(np.max(y_counts) * len(y_counts))

    return new_model, rows_after
