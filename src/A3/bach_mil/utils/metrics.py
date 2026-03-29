from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = y_prob.argmax(axis=1)
    out = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }
    try:
        out['ovr_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except Exception:
        out['ovr_auc'] = float('nan')
    return out


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    if thresholds is None:
        thresholds = np.full((y_prob.shape[1],), 0.5, dtype=np.float32)
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int64)
    out = {
        'sample_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'exact_match': float((y_true == y_pred).all(axis=1).mean()),
    }
    per_class_auc = []
    per_class_ap = []
    for i in range(y_true.shape[1]):
        try:
            per_class_auc.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
        except Exception:
            per_class_auc.append(float('nan'))
        try:
            per_class_ap.append(float(average_precision_score(y_true[:, i], y_prob[:, i])))
        except Exception:
            per_class_ap.append(float('nan'))
    out['per_class_auc'] = per_class_auc
    out['per_class_ap'] = per_class_ap
    out['macro_ap'] = float(np.nanmean(per_class_ap))
    return out


def search_thresholds(y_true: np.ndarray, y_prob: np.ndarray, num_steps: int = 81) -> np.ndarray:
    thresholds = np.zeros((y_prob.shape[1],), dtype=np.float32)
    cand = np.linspace(0.1, 0.9, num_steps)
    for c in range(y_prob.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        yt = y_true[:, c]
        yp = y_prob[:, c]
        # Degenerate validation label distributions are common with small slide counts.
        # Make threshold choice stable and conservative:
        # - all-negative: pick 1.0 so we do not emit false positives
        # - all-positive: pick 0.0 so we always emit the label
        if int(yt.sum()) == 0:
            thresholds[c] = 1.0
            continue
        if int(yt.sum()) == int(len(yt)):
            thresholds[c] = 0.0
            continue
        for t in cand:
            pred = (yp >= t).astype(np.int64)
            score = f1_score(yt, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresholds[c] = best_t
    return thresholds
