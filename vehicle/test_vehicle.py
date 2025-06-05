import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifiers_evaluation import analyze_param_impact

# Load vehicle data from CSV
df = pd.read_csv("vehicle/vehicles_data.csv")
X = df.drop(columns=["Class"]).values
y = df["Class"].values

# Encode class labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Fixed parameters for experiments
fixed_defaults = {
    'tree_percentage': 0.9,
    'trees_number': 21,
    'max_features': 'sqrt',
    'epsilon': 1.0,
    'test_size': 0.3
}
n_runs = 10

# === 1. Test: change tree_percentage ===
tree_percentages = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_percentages,
    param_name='tree_percentage',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'tree_percentage'},
    n_runs=n_runs
)

# === 2. Test: change trees_number ===
tree_counts = [11, 21, 30, 51, 80, 100]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_counts,
    param_name='trees_number',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'trees_number'},
    n_runs=n_runs
)

# === 3. Test: change max_features ===
feature_options = [1, "sqrt", 5 ,"all"]
analyze_param_impact(
    X=X,
    y=y,
    param_values=feature_options,
    param_name='max_features',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'max_features'},
    n_runs=n_runs
)

# === 4. Test: change epsilon ===
epsilon_values = [1e-10, 1e-6, 1e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=epsilon_values,
    param_name='epsilon',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'epsilon'},
    n_runs=n_runs
)
