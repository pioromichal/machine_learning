from plot_utils import plot_metric_lines_from_json, plot_confusion_from_json

plot_metric_lines_from_json(
    json_path="results/iris/results_tree_percentage_20250521_150847.json",
    # save_path="results/iris/metric_lines_tree_percentage.png",
    y_range=(0.8, 1.0)
)

plot_confusion_from_json(
    json_path="results/iris/results_tree_percentage_20250521_150847.json",
    label_value=0.5,
    # save_path="results/iris/confusion_0.5.png"
)


