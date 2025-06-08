import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker

PARAM_NAME_MAP = {
    'trees_number': 'Liczba drzew',
    'tree_percentage': 'Procent drzew',
    'max_features': 'Liczba cech',
    'epsilon': 'Epsilon'
}


def _get_json_path(param_name, index):
    return os.path.join("results", f"{param_name}_{index}.json")


def _get_plot_path(param_name, index, suffix):
    os.makedirs("plots", exist_ok=True)
    return os.path.join("plots", f"{param_name}_{index}_{suffix}.pdf")


def _polish_formatter(x, pos):
    return f"{round(x, 2):.2f}".replace('.', ',')


def generate_full_report(param_name: str, index: int = 1, y_range=(0.0, 1.0)):
    json_path = _get_json_path(param_name, index)
    if not os.path.exists(json_path):
        print(f"[BŁĄD] Nie znaleziono pliku: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    labels_str = list(data['results'].keys())

    if param_name == 'max_features':
        num_features = int(data['n_features'])
        labels = []
        for label in labels_str:
            if label == 'sqrt':
                labels.append(int(np.sqrt(num_features)))
            elif label == 'all':
                labels.append(num_features)
            else:
                labels.append(int(label))
    else:
        labels = [float(label) for label in labels_str]

    metrics_dicts = [data['results'][label]['avg_metrics'] for label in labels_str]
    df_avg = pd.DataFrame(metrics_dicts, index=labels)

    per_run_metrics = {}
    for label_str in labels_str:
        per_run_metrics[label_str] = data['results'][label_str]['per_run_metrics']

    param_label = PARAM_NAME_MAP.get(param_name, param_name)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Dokładność', 'Precyzja', 'Czułość', 'F1']

    for i, metric in enumerate(metrics):
        avg_vals = df_avg[metric]

        min_vals = []
        max_vals = []
        for label_str in labels_str:
            runs = [run[metric] for run in per_run_metrics[label_str]]
            min_vals.append(np.min(runs))
            max_vals.append(np.max(runs))

        axs[i].plot(labels, avg_vals, marker='o', label=metric_labels[i])
        axs[i].fill_between(labels, min_vals, max_vals, color='gray', alpha=0.2, label='Zakres min-maks')
        axs[i].set_ylim(*y_range)
        axs[i].set_ylabel(metric_labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='lower right')
        axs[i].yaxis.set_major_formatter(mticker.FuncFormatter(_polish_formatter))

    axs[-1].set_xlabel(param_label)
    axs[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_polish_formatter))
    fig.suptitle(f"Zależność metryk od parametru: {param_label}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(_get_plot_path(param_name, index, "subplot"))
    plt.close()

    for label_str, result in data['results'].items():
        cm = np.array(result['confusion_matrix'])
        num_classes = cm.shape[0]
        class_labels = [str(i) for i in range(num_classes)]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Przewidziana klasa")
        plt.ylabel("Rzeczywista klasa")
        plt.title(f"Macierz pomyłek: {param_label}={label_str}")
        plt.tight_layout()
        plt.savefig(_get_plot_path(param_name, index, f"confusion_{label_str}"))
        plt.close()
