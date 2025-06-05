import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_digits
from classifiers_evaluation import analyze_param_impact

# Wczytanie zbioru digits
digits = load_digits()
X = digits.data
y = digits.target

# --- Parametry stałe dla eksperymentów ---
fixed_defaults = {
    'tree_percentage': 0.9,
    'trees_number': 150,
    'max_features': 'sqrt',
    'epsilon': 1.0,
    'test_size': 0.3
}
n_runs = 10

# === 1. Test: zmiana tree_percentage ===
tree_percentages = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_percentages,
    param_name='tree_percentage',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'tree_percentage'},
    n_runs=n_runs
)

# === 2. Test: zmiana trees_number ===
tree_counts = [11, 21, 51, 80, 150]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_counts,
    param_name='trees_number',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'trees_number'},
    n_runs=n_runs
)

# === 3. Test: zmiana max_features ===
feature_options = [1, "sqrt", 8 ,"all"]
analyze_param_impact(
    X=X,
    y=y,
    param_values=feature_options,
    param_name='max_features',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'max_features'},
    n_runs=n_runs
)

# === 4. Test: zmiana epsilon ===
epsilon_values = [1e-10, 1e-6, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=epsilon_values,
    param_name='epsilon',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'epsilon'},
    n_runs=n_runs
)
