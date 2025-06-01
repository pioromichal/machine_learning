import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred, average='macro'):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def print_metrics(metrics_dict):
    for k, v in metrics_dict.items():
        print(f"{k.capitalize()}: {v:.3f}")


def get_confusion(y_true, y_pred, all_labels=None):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    if all_labels is None:
        all_labels = np.unique(np.concatenate([y_true, y_pred]))
    return confusion_matrix(y_true, y_pred, labels=all_labels)
