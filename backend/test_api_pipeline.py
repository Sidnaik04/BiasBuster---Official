"""Test /api/optimize endpoint with Pipeline support."""

import sys
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 80)
print("TESTING /api/optimize ENDPOINT WITH PIPELINE SUPPORT")
print("=" * 80)

# Test with your exact request
test_request = {
    "upload_id": 45,
    "target_column": "income",
    "sensitive_columns": ["gender"],
    "method": "optuna",
    "n_trials": 20,
    "cv_folds": 5,
    "accuracy_weight": 0.6,
    "fairness_weight": 0.4,
    "timeout": 300,
}

print("\n[Test 1] Testing /api/optimize with Optuna (Pipeline model)...")
print(f"Request body: {json.dumps(test_request, indent=2)}")

try:
    response = requests.post(f"{BASE_URL}/api/optimize", json=test_request, timeout=60)

    print(f"\nResponse Status: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print("✓ Request successful")

        if result.get("status") == "success":
            print(
                "✓ Optimization successful (no 'No Optuna parameters available for Pipeline' error)"
            )
            print(f"  - Best score: {result['result'].get('best_score', 'N/A')}")
            print(
                f"  - Combined score: {result['result'].get('combined_score', 'N/A')}"
            )
            print(f"  - Accuracy: {result['result'].get('accuracy', 'N/A')}")
            print(
                f"  - Fairness score: {result['result'].get('fairness_score', 'N/A')}"
            )
            print(f"  - DPD: {result['result'].get('dpd', 'N/A')}")
            print(f"  - EOD: {result['result'].get('eod', 'N/A')}")
            print(f"  - Trials run: {result['result'].get('trials_run', 'N/A')}")
            print(
                f"  - Best params keys: {list(result['result'].get('best_params', {}).keys())}"
            )
        else:
            print(
                f"✗ Optimization failed: {result['result'].get('message', 'Unknown error')}"
            )
    else:
        print(f"✗ Request failed with status {response.status_code}")

    print(f"\nFull response:")
    print(json.dumps(result, indent=2))

except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to server. Is it running on http://localhost:8000?")
except Exception as e:
    print(f"✗ Error: {str(e)}")

# Test with GridSearch
print("\n" + "=" * 80)
print("[Test 2] Testing /api/optimize with GridSearch (Pipeline model)...")

test_request_gs = test_request.copy()
test_request_gs["method"] = "gridsearch"
test_request_gs["n_trials"] = 5

print(f"Request body: {json.dumps(test_request_gs, indent=2)}")

try:
    response = requests.post(
        f"{BASE_URL}/api/optimize", json=test_request_gs, timeout=120
    )

    print(f"\nResponse Status: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print("✓ Request successful")

        if result.get("status") == "success":
            print(
                "✓ Optimization successful (no 'No parameter grid available for Pipeline' error)"
            )
            print(f"  - Best score: {result['result'].get('best_score', 'N/A')}")
            print(
                f"  - Combined score: {result['result'].get('combined_score', 'N/A')}"
            )
            print(f"  - Accuracy: {result['result'].get('accuracy', 'N/A')}")
            print(
                f"  - Fairness score: {result['result'].get('fairness_score', 'N/A')}"
            )
            print(f"  - Trials run: {result['result'].get('trials_run', 'N/A')}")
            print(
                f"  - Best params keys: {list(result['result'].get('best_params', {}).keys())}"
            )
        else:
            print(
                f"✗ Optimization failed: {result['result'].get('message', 'Unknown error')}"
            )
    else:
        print(f"✗ Request failed with status {response.status_code}")

    print(f"\nFull response:")
    print(json.dumps(result, indent=2))

except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to server. Is it running on http://localhost:8000?")
except Exception as e:
    print(f"✗ Error: {str(e)}")

print("\n" + "=" * 80)
print("ENDPOINT TESTS COMPLETED")
print("=" * 80)
