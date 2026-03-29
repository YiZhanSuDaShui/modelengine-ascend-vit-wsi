from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import save_json
from bach_mil.utils.metrics import search_thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_csv', type=str, required=True)
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', required=True)
    parser.add_argument('--all_positive_threshold', type=float, default=0.5)
    parser.add_argument('--all_negative_threshold', type=float, default=1.0)
    args = parser.parse_args()

    label_names = list(args.label_names)
    probs = pd.read_csv(args.prob_csv)
    bags = pd.read_csv(args.bag_csv)

    for name in label_names:
        pc = f'prob_{name}'
        lc = f'label_{name}'
        if pc not in probs.columns:
            raise SystemExit(f'prob_csv missing column {pc}')
        if lc not in bags.columns:
            raise SystemExit(f'bag_csv missing column {lc}')

    df = bags.merge(probs[['slide_id'] + [f'prob_{n}' for n in label_names]], on='slide_id', how='inner')
    if df.empty:
        raise SystemExit('no overlapping slide_id between prob_csv and bag_csv')

    y_true = df[[f'label_{n}' for n in label_names]].values.astype(np.int64)
    y_prob = df[[f'prob_{n}' for n in label_names]].values.astype(np.float32)

    thr = search_thresholds(y_true, y_prob)
    for i, name in enumerate(label_names):
        s = int(y_true[:, i].sum())
        if s == 0:
            thr[i] = float(args.all_negative_threshold)
        elif s == int(len(y_true)):
            thr[i] = float(args.all_positive_threshold)

    out = {name: float(thr[i]) for i, name in enumerate(label_names)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(out, args.out_json)
    print(f'saved -> {args.out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
