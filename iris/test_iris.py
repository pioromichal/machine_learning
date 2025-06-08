from sklearn.datasets import load_iris
from classifiers_evaluation import analyze_param_impact

iris = load_iris()
X = iris.data
y = iris.target

fixed_defaults = {
    'tree_percentage': 0.5,
    'trees_number': 21,
    'max_features': 'sqrt',
    'epsilon': 1e-6,
    'test_size': 0.3
}
n_runs = 30

tree_counts = [1, 25, 50, 75, 90, 100, 110, 125, 150]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_counts,
    param_name='trees_number',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'trees_number'},
    n_runs=n_runs
)

fixed_defaults['trees_number'] = 100

tree_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=tree_percentages,
    param_name='tree_percentage',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'tree_percentage'},
    n_runs=n_runs
)

fixed_defaults['tree_percentage'] = 0.5


feature_options = [1, "sqrt", 3, "all"]
analyze_param_impact(
    X=X,
    y=y,
    param_values=feature_options,
    param_name='max_features',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'max_features'},
    n_runs=n_runs
)

fixed_defaults['max_features'] = "sqrt"

epsilon_values = [1e-6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
analyze_param_impact(
    X=X,
    y=y,
    param_values=epsilon_values,
    param_name='epsilon',
    fixed_params={k: v for k, v in fixed_defaults.items() if k != 'epsilon'},
    n_runs=n_runs
)

