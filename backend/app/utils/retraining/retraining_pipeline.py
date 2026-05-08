"""Retraining pipeline for post-mitigation models.

Implements retraining for 'reweighting' and 'smote' strategies, evaluates
classification and fairness metrics, versions models, and stores metadata.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from app.config import settings
from app.utils.fairness import metrics as fairness_metrics
from app.utils.mitigation.smote import apply_smote

ARTIFACTS_ROOT_DIR = Path(settings.ARTIFACT_DIR)
ARTIFACTS_MODELS_DIR = ARTIFACTS_ROOT_DIR / "models"
ARTIFACTS_RETRAINING_DIR = ARTIFACTS_ROOT_DIR / "retraining"
ARTIFACTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_RETRAINING_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RetrainingResult:
    retraining_id: str
    strategy_used: str
    original_model_metrics: Dict[str, Any]
    mitigated_model_metrics: Dict[str, Any]
    retrained_model_metrics: Dict[str, Any]
    fairness_stability: Dict[str, Any]
    model_version: str
    training_metadata: Dict[str, Any]
    summary: str


def _safe_roc_auc(y_true, y_score) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _normalize_binary_label_types(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize y_true/y_pred types for binary metrics.

    Handles common mismatch cases such as string ground truth labels with numeric
    predictions (0/1). Returns arrays with aligned, comparable label types.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Quick path: same dtype and numeric labels
    if y_true_arr.dtype == y_pred_arr.dtype:
        if np.issubdtype(y_true_arr.dtype, np.number):
            return y_true_arr, y_pred_arr

        # If both are binary non-numeric labels, map consistently to 0/1.
        union_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        if len(union_labels) == 2:
            ordered = sorted(union_labels, key=lambda x: str(x))
            mapping = {ordered[0]: 0, ordered[1]: 1}
            return (
                np.array([mapping[v] for v in y_true_arr]).astype(int),
                np.array([mapping[v] for v in y_pred_arr]).astype(int),
            )

        return y_true_arr, y_pred_arr

    u_true = np.unique(y_true_arr)
    u_pred = np.unique(y_pred_arr)

    # Case 1: y_pred is binary numeric and y_true has 2 non-numeric labels
    if len(u_true) == 2 and set(np.asarray(u_pred).tolist()).issubset({0, 1}):
        if not np.issubdtype(y_true_arr.dtype, np.number):
            ordered = sorted(u_true, key=lambda x: str(x))
            mapping = {ordered[0]: 0, ordered[1]: 1}
            y_true_mapped = np.array([mapping[v] for v in y_true_arr])
            return y_true_mapped.astype(int), y_pred_arr.astype(int)

    # Case 2: y_true is binary numeric and y_pred has 2 non-numeric labels
    if len(u_pred) == 2 and set(np.asarray(u_true).tolist()).issubset({0, 1}):
        if not np.issubdtype(y_pred_arr.dtype, np.number):
            ordered = sorted(u_pred, key=lambda x: str(x))
            mapping = {ordered[0]: 0, ordered[1]: 1}
            y_pred_mapped = np.array([mapping[v] for v in y_pred_arr])
            return y_true_arr.astype(int), y_pred_mapped.astype(int)

    # Case 3: same label set but different primitive types
    true_as_str = np.array([str(v) for v in y_true_arr])
    pred_as_str = np.array([str(v) for v in y_pred_arr])
    if set(np.unique(true_as_str)) == set(np.unique(pred_as_str)):
        labels = sorted(np.unique(np.concatenate([true_as_str, pred_as_str])))
        mapping = {label: idx for idx, label in enumerate(labels)}
        y_true_mapped = np.array([mapping[v] for v in true_as_str])
        y_pred_mapped = np.array([mapping[v] for v in pred_as_str])
        return y_true_mapped, y_pred_mapped

    # Last resort: stringify both to allow exact-match metrics like accuracy.
    return true_as_str, pred_as_str


def compute_classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, Any]:
    y_true, y_pred = _normalize_binary_label_types(y_true, y_pred)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    unique_classes = np.unique(y_true)
    average = "binary" if len(unique_classes) == 2 else "weighted"
    metric_kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        # Ensure pos_label matches existing class value when labels are not {0,1}.
        metric_kwargs["pos_label"] = sorted(unique_classes, key=lambda x: str(x))[-1]

    metrics["precision"] = float(precision_score(y_true, y_pred, **metric_kwargs))
    metrics["recall"] = float(recall_score(y_true, y_pred, **metric_kwargs))
    metrics["f1"] = float(f1_score(y_true, y_pred, **metric_kwargs))
    if y_score is not None:
        # roc_auc is only valid for numeric binary labels in this pipeline.
        if len(unique_classes) == 2 and np.issubdtype(
            np.asarray(y_true).dtype, np.number
        ):
            auc = _safe_roc_auc(y_true, y_score)
            if auc is not None:
                metrics["roc_auc"] = float(auc)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def compute_fairness_metrics(
    dataset: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    # Expect binary preds (0/1)
    dpd_list = []
    eod_list = []
    di_list = []

    for attr_col in dataset.select_dtypes(include=[object, "category"]).columns:
        try:
            groups = dataset[attr_col].unique()
            if len(groups) < 2:
                continue
            g0 = groups[0]
            g1 = groups[1]
            mask0 = dataset[attr_col] == g0
            mask1 = dataset[attr_col] == g1

            sr0 = fairness_metrics.selection_rate(y_pred[mask0])
            sr1 = fairness_metrics.selection_rate(y_pred[mask1])
            dpd = abs(sr0 - sr1)
            dpd_list.append(dpd)

            if y_true[mask0].sum() > 0 and y_true[mask1].sum() > 0:
                tpr0 = fairness_metrics.true_positive_rate(y_true[mask0], y_pred[mask0])
                tpr1 = fairness_metrics.true_positive_rate(y_true[mask1], y_pred[mask1])
                eod = abs(tpr0 - tpr1)
                eod_list.append(eod)

            # DI (disparate impact ratio)
            rates = {g0: sr0, g1: sr1}
            di = fairness_metrics.disparate_impact_ratio(rates)
            di_list.append(di)

        except Exception:
            continue

    dpd = float(np.mean(dpd_list)) if dpd_list else 0.0
    eod = float(np.mean(eod_list)) if eod_list else 0.0
    di = float(np.mean(di_list)) if di_list else 0.0
    fairness_score = 1.0 - (abs(dpd) + abs(eod)) / 2.0

    return {"dpd": dpd, "eod": eod, "di": di, "fairness_score": fairness_score}


def _evaluate_model(
    model, dataset: pd.DataFrame, target_column: str
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    X = dataset.drop(columns=[target_column], errors="ignore")
    y = dataset[target_column].values

    # Predict
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)[:, 1]
        except Exception:
            y_score = None
    else:
        y_score = None

    try:
        y_pred = model.predict(X)
    except Exception:
        # fallback: thresholded probabilities if available
        if y_score is not None:
            y_pred = (np.array(y_score) >= 0.5).astype(int)
        else:
            y_pred = np.zeros_like(y)

    y_eval_true, y_eval_pred = _normalize_binary_label_types(y, y_pred)

    class_metrics = compute_classification_metrics(y_eval_true, y_eval_pred, y_score)
    fairness = compute_fairness_metrics(dataset, y_eval_true, y_eval_pred)
    combined = {**class_metrics, **fairness}
    return combined, y_eval_pred, y_score


def _train_model_clone(
    base_model, X_train: pd.DataFrame, y_train: pd.Series, sample_weight=None
):
    # Try cloning and fitting uploaded model, else return None
    from sklearn.base import clone

    try:
        m = clone(base_model)
        if sample_weight is not None and "fit" in dir(m):
            try:
                m.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                m.fit(X_train, y_train)
        else:
            m.fit(X_train, y_train)
        return m
    except Exception:
        return None


def _compute_fairness_reweighting_weights(
    df: pd.DataFrame, target_column: str, sensitive_columns: list
) -> np.ndarray:
    """Compute Kamiran-style reweighting weights.

    weight(s, y) = P(s) * P(y) / P(s, y)
    where s is the (possibly intersectional) sensitive group.
    """
    if not sensitive_columns:
        return np.ones(len(df), dtype=float)

    valid_sensitive = [c for c in sensitive_columns if c in df.columns]
    if not valid_sensitive:
        return np.ones(len(df), dtype=float)

    work = df.copy()
    group_key = work[valid_sensitive].astype(str).agg("|".join, axis=1)
    y = work[target_column].astype(str)
    n = len(work)

    p_s = group_key.value_counts(dropna=False) / n
    p_y = y.value_counts(dropna=False) / n
    p_sy = (
        pd.DataFrame({"s": group_key, "y": y}).value_counts(dropna=False).rename("p")
        / n
    )

    weights = np.ones(n, dtype=float)
    for idx, (s_val, y_val) in enumerate(zip(group_key.values, y.values)):
        joint = p_sy.get((s_val, y_val), 0.0)
        if joint > 0:
            weights[idx] = float((p_s.get(s_val, 0.0) * p_y.get(y_val, 0.0)) / joint)

    # Normalize around mean=1 to keep optimizer scale stable.
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def run_retraining(
    upload_dataset: pd.DataFrame,
    uploaded_model,
    target_column: str,
    sensitive_columns: list,
    strategy: str,
    train_additional_models: bool = True,
    random_seed: int = 42,
    test_size: float = 0.2,
) -> RetrainingResult:
    """Main entry for retraining pipeline.

    Returns a RetrainingResult dataclass instance.
    """
    retraining_id = f"rt_{uuid.uuid4().hex[:8]}"

    # Save seeds and metadata
    np.random.seed(random_seed)

    # Stage 0: create baseline split and evaluate original model on holdout.
    base_train, base_test = train_test_split(
        upload_dataset,
        test_size=test_size,
        random_state=random_seed,
        stratify=upload_dataset[target_column],
    )
    comparison_eval_dataset = base_test.copy()
    orig_metrics, orig_pred, orig_score = _evaluate_model(
        uploaded_model, comparison_eval_dataset, target_column
    )

    # Stage 1: train a true mitigated model on base_train.
    corrected_dataset = upload_dataset.copy()
    X_base_train = base_train.drop(columns=[target_column], errors="ignore")
    y_base_train = base_train[target_column]
    mitigated_model = None
    mitigation_warnings = []

    if strategy == "reweighting":
        valid_sensitive = [c for c in sensitive_columns if c in base_train.columns]
        if not valid_sensitive:
            mitigation_warnings.append(
                "Sensitive column unavailable for reweighting; used original model."
            )
            mitigated_model = uploaded_model
        else:
            try:
                sample_weights = _compute_fairness_reweighting_weights(
                    base_train,
                    target_column,
                    valid_sensitive,
                )
                mitigated_model = _train_model_clone(
                    uploaded_model,
                    X_base_train,
                    y_base_train,
                    sample_weight=sample_weights,
                )
                if mitigated_model is None:
                    mitigation_warnings.append(
                        "Reweighting mitigation retraining failed; fallback to original model."
                    )
            except Exception as exc:
                mitigation_warnings.append(f"Reweighting mitigation failed: {exc}")
                mitigated_model = None
    elif strategy == "smote":
        sensitive_col = sensitive_columns[0] if sensitive_columns else None
        sensitive_train = (
            base_train[sensitive_col]
            if sensitive_col is not None and sensitive_col in base_train.columns
            else np.zeros(len(base_train), dtype=int)
        )
        try:
            mitigated_model, _ = apply_smote(
                X_base_train,
                y_base_train,
                sensitive_train,
                uploaded_model,
            )
        except Exception as exc:
            mitigation_warnings.append(f"SMOTE mitigation failed: {exc}")
            mitigated_model = None
    else:
        raise ValueError("Only 'reweighting' and 'smote' supported for retraining")

    if mitigated_model is None:
        mitigated_model = uploaded_model

    mitigated_metrics, mitigated_pred, mitigated_score = _evaluate_model(
        mitigated_model, comparison_eval_dataset, target_column
    )

    # Stage 2: generate NEW train/test split for retraining.
    retrain_split_seed = random_seed + 1
    train, test = train_test_split(
        corrected_dataset,
        test_size=test_size,
        random_state=retrain_split_seed,
        stratify=corrected_dataset[target_column],
    )
    X_train = train.drop(columns=[target_column], errors="ignore")
    y_train = train[target_column]

    # Compute sample weights if reweighting
    sample_weights = None
    if strategy == "reweighting":
        valid_sensitive = [c for c in sensitive_columns if c in train.columns]
        if valid_sensitive:
            sample_weights = _compute_fairness_reweighting_weights(
                train,
                target_column,
                valid_sensitive,
            )

    # Retrain uploaded model
    retrained_uploaded = None
    if strategy == "smote":
        sensitive_col = sensitive_columns[0] if sensitive_columns else None
        sensitive_train = (
            train[sensitive_col]
            if sensitive_col is not None and sensitive_col in train.columns
            else np.zeros(len(train), dtype=int)
        )
        try:
            retrained_uploaded, _ = apply_smote(
                X_train, y_train, sensitive_train, uploaded_model
            )
        except Exception as exc:
            mitigation_warnings.append(f"SMOTE retraining failed: {exc}")
            retrained_uploaded = None
    else:
        retrained_uploaded = _train_model_clone(
            uploaded_model, X_train, y_train, sample_weight=sample_weights
        )
        if retrained_uploaded is None:
            mitigation_warnings.append(
                "Reweighting retraining failed; fallback to original model."
            )

    # Train benchmark models optionally
    benchmark_models = {}
    if train_additional_models:
        try:
            benchmark_models["logistic_regression"] = LogisticRegression(max_iter=1000)
            benchmark_models["random_forest"] = RandomForestClassifier(
                n_estimators=100, random_state=random_seed
            )
            benchmark_models["gradient_boosting"] = GradientBoostingClassifier(
                random_state=random_seed
            )
        except Exception:
            benchmark_models = {}

    trained_models = {}
    # Add retrained uploaded model (fallback to original if retraining failed)
    trained_models["uploaded_model"] = retrained_uploaded or uploaded_model

    # Train benchmarks
    for name, model in benchmark_models.items():
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception:
            continue

    # Evaluate retrained uploaded model on the same comparison holdout.
    final_model = trained_models.get("uploaded_model")
    final_test_dataset = comparison_eval_dataset
    retrained_metrics, retrained_pred, retrained_score = _evaluate_model(
        final_model, final_test_dataset, target_column
    )

    # Also evaluate best benchmark model by fairness+accuracy heuristic
    best_benchmark_name = None
    best_benchmark_score = -1
    for name, model in trained_models.items():
        if name == "uploaded_model":
            continue
        m_metrics, _, _ = _evaluate_model(model, final_test_dataset, target_column)
        # simple heuristic: prioritize fairness_score then accuracy
        score_val = (
            m_metrics.get("fairness_score", 0) * 0.7
            + m_metrics.get("accuracy", 0) * 0.3
        )
        if score_val > best_benchmark_score:
            best_benchmark_score = score_val
            best_benchmark_name = name

    # Choose best final model between retrained uploaded and best benchmark
    chosen_model = final_model
    chosen_name = "uploaded_model"
    if best_benchmark_name:
        bench_metrics, _, _ = _evaluate_model(
            trained_models[best_benchmark_name], final_test_dataset, target_column
        )
        bench_score_val = (
            bench_metrics.get("fairness_score", 0) * 0.7
            + bench_metrics.get("accuracy", 0) * 0.3
        )
        retrained_score_val = (
            retrained_metrics.get("fairness_score", 0) * 0.7
            + retrained_metrics.get("accuracy", 0) * 0.3
        )
        if bench_score_val > retrained_score_val:
            chosen_model = trained_models[best_benchmark_name]
            chosen_name = best_benchmark_name

    # Stage 3: evaluate chosen model on the same holdout for apples-to-apples reporting
    final_metrics, final_pred, final_score = _evaluate_model(
        chosen_model, comparison_eval_dataset, target_column
    )

    # Fairness stability check.
    def _significant_regression(orig, new, tol=0.02):
        # significant if increase (worse) by more than tol
        return (new - orig) > tol

    stable = True
    observations = []

    tol = 0.02
    mitigated_gain = mitigated_metrics.get("fairness_score", 0) - orig_metrics.get(
        "fairness_score", 0
    )
    retrained_gain = final_metrics.get("fairness_score", 0) - orig_metrics.get(
        "fairness_score", 0
    )

    if mitigated_gain <= tol:
        stable = False
        observations.append(
            "Mitigation stage did not improve fairness over original baseline"
        )

    if _significant_regression(
        mitigated_metrics.get("dpd", 0), final_metrics.get("dpd", 0), tol=tol
    ):
        stable = False
        observations.append("DPD regressed significantly after retraining")
    if _significant_regression(
        mitigated_metrics.get("eod", 0), final_metrics.get("eod", 0), tol=tol
    ):
        stable = False
        observations.append("EOD regressed significantly after retraining")

    if retrained_gain <= tol:
        stable = False
        observations.append(
            "Retrained model does not retain fairness improvement over original"
        )

    if stable:
        observations.append("Fairness improvements remained stable after retraining.")
    else:
        observations.append("Fairness stability check failed; review strategy/model.")

    # Model versioning
    version_suffix = f"{uuid.uuid4().hex[:6]}"
    model_version_name = f"model_{version_suffix}_{strategy}_retrained.joblib"
    model_version_path = ARTIFACTS_MODELS_DIR / model_version_name

    try:
        joblib.dump(chosen_model, model_version_path)
    except Exception:
        # fallback: try to save uploaded model
        joblib.dump(
            uploaded_model,
            ARTIFACTS_MODELS_DIR / f"model_{version_suffix}_fallback.joblib",
        )
        model_version_name = f"model_{version_suffix}_fallback.joblib"

    # Metadata
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_shape": list(corrected_dataset.shape),
        "strategy": strategy,
        "train_test_split_seed": int(random_seed),
        "retrain_split_seed": int(retrain_split_seed),
        "hyperparameters": {},
        "model_type": chosen_name,
        "warnings": mitigation_warnings,
        "comparison": {
            "mitigated_fairness_gain": float(mitigated_gain),
            "retrained_fairness_gain": float(retrained_gain),
            "accuracy_recovery": float(
                final_metrics.get("accuracy", 0) - mitigated_metrics.get("accuracy", 0)
            ),
        },
    }

    metadata_path = ARTIFACTS_RETRAINING_DIR / f"{retraining_id}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {
                **metadata,
                "original_metrics": orig_metrics,
                "mitigated_metrics": mitigated_metrics,
                "final_metrics": final_metrics,
            },
            f,
            indent=2,
        )

    if stable:
        summary = "Retraining preserved fairness gains over baseline on a consistent holdout evaluation."
    else:
        summary = (
            "Fairness-aware retraining did not produce stable gains over baseline; "
            "consider changing strategy, sensitive feature choice, or model class."
        )

    result = RetrainingResult(
        retraining_id=retraining_id,
        strategy_used=strategy,
        original_model_metrics=orig_metrics,
        mitigated_model_metrics=mitigated_metrics,
        retrained_model_metrics=final_metrics,
        fairness_stability={"stable": stable, "observation": " ".join(observations)},
        model_version=str(model_version_name),
        training_metadata=metadata,
        summary=summary,
    )

    return result
