"""Test Pipeline support for model optimizer."""

import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add app to path
sys.path.insert(0, "/home/sidnaik04/Documents/BiasBuster/BiasBuster - Code/backend")

from app.utils.optimization.model_optimizer import (
    FairnessAwareOptimizer,
    get_model_class_name,
    extract_base_estimator,
    get_param_grid,
    get_optuna_params,
)

print("=" * 80)
print("TESTING PIPELINE SUPPORT FOR MODEL OPTIMIZER")
print("=" * 80)

# ============================================================================
# TEST 1: Helper functions
# ============================================================================
print("\n[TEST 1] Testing helper functions...")

# Create a Pipeline
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("estimator", LogisticRegression(max_iter=200))]
)

# Test extract_base_estimator
base = extract_base_estimator(pipeline)
print(f"✓ extract_base_estimator works: {base.__class__.__name__}")

# Test get_model_class_name with Pipeline
model_name = get_model_class_name(pipeline)
print(f"✓ get_model_class_name for Pipeline: {model_name}")
assert (
    model_name == "LogisticRegression"
), f"Expected LogisticRegression, got {model_name}"

# Test get_model_class_name with direct model
direct_model = LogisticRegression()
model_name_direct = get_model_class_name(direct_model)
print(f"✓ get_model_class_name for direct model: {model_name_direct}")
assert (
    model_name_direct == "LogisticRegression"
), f"Expected LogisticRegression, got {model_name_direct}"

# ============================================================================
# TEST 2: Parameter grids for Pipeline extractors
# ============================================================================
print("\n[TEST 2] Testing parameter grid retrieval...")

# Get param grid for LogisticRegression (from Pipeline)
param_grid_lr = get_param_grid("LogisticRegression")
print(f"✓ LogisticRegression PARAM_GRID has {len(param_grid_lr)} parameters")

# Get optuna params
optuna_params_lr = get_optuna_params("LogisticRegression")
print(f"✓ LogisticRegression OPTUNA_PARAMS has {len(optuna_params_lr)} parameters")

# ============================================================================
# TEST 3: Optimizer with Pipeline
# ============================================================================
print("\n[TEST 3] Testing FairnessAwareOptimizer with Pipeline...")

# Create synthetic data
np.random.seed(42)
n_samples = 200

# Features
X = np.random.randn(n_samples, 5)

# Target (binary)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Sensitive attribute (binary)
sensitive = pd.DataFrame({"gender": np.random.choice(["M", "F"], n_samples)})

# Create Pipeline models
pipeline_lr = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("estimator", LogisticRegression(max_iter=500, random_state=42)),
    ]
)

pipeline_rf = Pipeline(
    [
        ("preprocessor", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=50, random_state=42)),
    ]
)

# Test with LogisticRegression Pipeline
print("\n  Testing with LogisticRegression Pipeline...")
try:
    optimizer_lr = FairnessAwareOptimizer(
        model=pipeline_lr,
        X=X,
        y=y,
        sensitive_features=sensitive,
        accuracy_weight=0.6,
        fairness_weight=0.4,
    )
    print(f"  ✓ Optimizer created successfully for LogisticRegression Pipeline")

    # Get parameter grid
    model_name = get_model_class_name(pipeline_lr)
    param_grid = get_param_grid(model_name)
    print(f"  ✓ Parameter grid retrieved: {model_name} ({len(param_grid)} params)")

    # Simplified grid for faster testing
    simplified_grid = {
        "C": [0.1, 1],
        "max_iter": [100, 200],
    }

    print(f"  ⏳ Running GridSearchCV (simplified)...")
    result_gs = optimizer_lr.optimize_with_gridsearch(param_grid=simplified_grid, cv=2)

    print(f"  ✓ GridSearchCV completed")
    print(f"     - Best score: {result_gs['combined_score']:.4f}")
    print(f"     - Accuracy: {result_gs['accuracy']:.4f}")
    print(f"     - Fairness score: {result_gs['fairness_score']:.4f}")
    print(f"     - Trials run: {result_gs['trials_run']}")

except Exception as e:
    print(f"  ✗ Error with LogisticRegression Pipeline: {str(e)}")
    import traceback

    traceback.print_exc()

# Test with RandomForest Pipeline
print("\n  Testing with RandomForest Pipeline...")
try:
    optimizer_rf = FairnessAwareOptimizer(
        model=pipeline_rf,
        X=X,
        y=y,
        sensitive_features=sensitive,
        accuracy_weight=0.6,
        fairness_weight=0.4,
    )
    print(f"  ✓ Optimizer created successfully for RandomForest Pipeline")

    # Get parameter grid
    model_name = get_model_class_name(pipeline_rf)
    param_grid = get_param_grid(model_name)
    print(f"  ✓ Parameter grid retrieved: {model_name} ({len(param_grid)} params)")

    # Simplified grid for faster testing
    simplified_grid = {
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    }

    print(f"  ⏳ Running GridSearchCV (simplified)...")
    result_gs = optimizer_rf.optimize_with_gridsearch(param_grid=simplified_grid, cv=2)

    print(f"  ✓ GridSearchCV completed")
    print(f"     - Best score: {result_gs['combined_score']:.4f}")
    print(f"     - Accuracy: {result_gs['accuracy']:.4f}")
    print(f"     - Fairness score: {result_gs['fairness_score']:.4f}")
    print(f"     - Trials run: {result_gs['trials_run']}")

except Exception as e:
    print(f"  ✗ Error with RandomForest Pipeline: {str(e)}")
    import traceback

    traceback.print_exc()

# ============================================================================
# TEST 4: Optuna with Pipeline
# ============================================================================
print("\n[TEST 4] Testing Optuna with Pipeline...")

try:
    optimizer_optuna = FairnessAwareOptimizer(
        model=pipeline_lr,
        X=X,
        y=y,
        sensitive_features=sensitive,
        accuracy_weight=0.6,
        fairness_weight=0.4,
    )

    model_name = get_model_class_name(pipeline_lr)
    optuna_params = get_optuna_params(model_name)

    # Simplified params for faster testing
    simplified_optuna = {
        "C": {"type": "float", "low": 0.1, "high": 10},
        "max_iter": {"type": "int", "low": 100, "high": 500},
    }

    print(f"  ⏳ Running Optuna optimization (5 trials)...")
    result_optuna = optimizer_optuna.optimize_with_optuna(
        param_distributions=simplified_optuna,
        n_trials=5,
        timeout=60,
    )

    print(f"  ✓ Optuna completed")
    print(f"     - Best score: {result_optuna['combined_score']:.4f}")
    print(f"     - Accuracy: {result_optuna.get('accuracy', 'N/A')}")
    print(f"     - Fairness score: {result_optuna.get('fairness_score', 'N/A')}")
    print(f"     - Trials run: {result_optuna['trials_run']}")

except ImportError:
    print(f"  ⓘ Optuna not installed, skipping test")
except Exception as e:
    print(f"  ✗ Error with Optuna: {str(e)}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("PIPELINE SUPPORT TEST COMPLETED")
print("=" * 80)
