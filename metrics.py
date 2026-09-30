import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def compute_metrics(y_true, fake_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    fake_prob = np.asarray(fake_prob, dtype=np.float64)
    pred = (fake_prob >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, pred))
    auc = float(roc_auc_score(y_true, fake_prob)) if len(np.unique(y_true)) == 2 else float("nan")

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, fake_prob)
        fnr = 1.0 - tpr
        eer = float(fpr[np.nanargmin(np.abs(fpr - fnr))])
    else:
        eer = float("nan")
    return {"acc": acc, "auc": auc, "eer": eer}
