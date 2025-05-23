import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from metrics_utils import compute_metrics, print_metrics, get_confusion
from classifiers import RandomForest


def evaluate_random_forest_avg(
    X: np.ndarray,
    y: np.ndarray,
    trees_number: int,
    tree_percentage: float,
    max_features="sqrt",
    epsilon=1e-6,
    n_runs: int = 10,
    test_size: float = 0.3,
    verbose: bool = True
):
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    confusion_matrices = []
    per_run_metrics = []
    per_run_confusions = []

    for i in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
        model = RandomForest(
            trees_number=trees_number,
            tree_percentage=tree_percentage,
            n_jobs=-1,
            max_features=max_features,
            epsilon=epsilon
        )
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = compute_metrics(y_test, preds)
            cm = get_confusion(y_test, preds)

            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            confusion_matrices.append(cm)
            per_run_metrics.append(metrics)
            per_run_confusions.append(cm.tolist())

            if verbose:
                print(f"Run {i+1}: ", end="")
                print_metrics(metrics)
        except Exception as e:
            print(f"Run {i+1} failed: {e}")

    if not all_metrics['accuracy']:
        raise RuntimeError("All runs failed.")

    avg_metrics = {key: np.mean(vals) for key, vals in all_metrics.items()}
    avg_confusion = np.mean(confusion_matrices, axis=0)
    print("\n=== AVERAGED METRICS ===")
    print_metrics(avg_metrics)

    return avg_metrics, avg_confusion, per_run_metrics, per_run_confusions


def analyze_param_impact(X, y, param_values, param_name='trees_number', fixed_params=None, n_runs=10):
    fixed_params = fixed_params or {}
    results = []
    labels = []
    confusions = []
    all_details = {
        "param_name": param_name,
        "param_values": param_values,
        "n_runs": n_runs,
        "fixed_params": fixed_params,
        "results": {}
    }

    for val in param_values:
        kwargs = fixed_params.copy()
        kwargs[param_name] = val
        label = str(val)

        avg_metrics, cm, per_run_metrics, per_run_confusions = evaluate_random_forest_avg(
            X=X,
            y=y,
            trees_number=kwargs.get('trees_number', 10),
            tree_percentage=kwargs.get('tree_percentage', 0.5),
            max_features=kwargs.get('max_features', "sqrt"),
            epsilon=kwargs.get('epsilon', 1e-6),
            n_runs=n_runs,
            test_size=kwargs.get('test_size', 0.3),
            verbose=False
        )

        results.append(avg_metrics)
        labels.append(label)
        confusions.append(cm)
        all_details["results"][label] = {
            "param_value": val,
            "full_params": kwargs,
            "avg_metrics": avg_metrics,
            "per_run_metrics": per_run_metrics,
            "per_run_confusions": per_run_confusions,
            "confusion_matrix": cm.tolist()
        }

    # === Zapis do results/ ===
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # Znajdź kolejny dostępny numer
    existing = [f for f in os.listdir(save_dir) if f.startswith(param_name) and f.endswith(".json")]
    existing_ids = [
        int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()
    ]
    next_id = max(existing_ids, default=0) + 1

    # Nazwa pliku
    json_filename = f"{param_name}_{next_id}.json"
    json_path = os.path.join(save_dir, json_filename)

    # Zapisz
    with open(json_path, 'w') as f:
        json.dump(all_details, f, indent=4)

    return results, labels, confusions, all_details
