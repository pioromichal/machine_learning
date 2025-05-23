import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker


def _get_json_path(param_name, index):
    return os.path.join("results", f"{param_name}_{index}.json")


def _get_plot_path(param_name, index, suffix):
    os.makedirs("plots", exist_ok=True)
    return os.path.join("plots", f"{param_name}_{index}_{suffix}.png")


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
    labels = [float(label) for label in labels_str]
    metrics_dicts = [data['results'][label]['avg_metrics'] for label in labels_str]
    df = pd.DataFrame(metrics_dicts, index=labels)

    # === 1. Wykres metryk 4x1 ===
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Dokładność', 'Precyzja', 'Czułość', 'F1']

    for i, metric in enumerate(metrics):
        axs[i].plot(labels, df[metric], marker='o', label=metric_labels[i])
        axs[i].set_ylim(*y_range)
        axs[i].set_ylabel(metric_labels[i])
        axs[i].grid(True)
        axs[i].legend(loc='lower right')
        axs[i].yaxis.set_major_formatter(mticker.FuncFormatter(_polish_formatter))

    axs[-1].set_xlabel(param_name.replace('_', ' ').capitalize())
    axs[-1].xaxis.set_major_formatter(mticker.FuncFormatter(_polish_formatter))
    fig.suptitle(f"Zależność metryk od parametru: {param_name.replace('_', ' ')}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(_get_plot_path(param_name, index, "subplot"))
    plt.close()

    # === 2. Sumaryczna macierz pomyłek ===
    conf_list = [np.array(res['confusion_matrix']) for res in data['results'].values()]
    sum_cm = np.sum(conf_list, axis=0)
    num_classes = sum_cm.shape[0]
    class_labels = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(6, 5))
    sns.heatmap(sum_cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")
    plt.title("Sumaryczna macierz pomyłek")
    plt.tight_layout()
    plt.savefig(_get_plot_path(param_name, index, f"confusion_sum"))
    plt.close()
