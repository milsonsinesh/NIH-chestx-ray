# src/metrics.py

from sklearn.metrics import roc_auc_score

def mean_auroc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="macro")
