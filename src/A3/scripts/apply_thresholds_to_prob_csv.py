from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prob_csv', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=None)
    parser.add_argument('--thresholds_json', type=str, default=None)
    parser.add_argument('--default_threshold', type=float, default=0.5)
    parser.add_argument('--normal_fallback', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.prob_csv)
    if 'slide_id' not in df.columns:
        raise SystemExit('prob_csv missing slide_id column')

    if args.label_names is not None:
        label_names = list(args.label_names)
    else:
        label_names = sorted([c[len('prob_'):] for c in df.columns if c.startswith('prob_')])
    if not label_names:
        raise SystemExit('no label_names inferred from prob_ columns')

    for name in label_names:
        col = f'prob_{name}'
        if col not in df.columns:
            raise SystemExit(f'prob_csv missing column {col}')

    thresholds = {}
    if args.thresholds_json is not None:
        thresholds = load_json(args.thresholds_json)

    out_rows = []
    for _, row in df.iterrows():
        active = []
        pred = {}
        for name in label_names:
            thr = float(thresholds.get(name, args.default_threshold))
            flag = int(float(row[f'prob_{name}']) >= thr)
            pred[name] = flag
            if flag == 1:
                active.append(name)
        if args.normal_fallback and ('Normal' in label_names) and len(active) == 0:
            active = ['Normal']
            pred['Normal'] = 1
        out = {'slide_id': row['slide_id'], 'pred_labels': ';'.join(active)}
        for name in label_names:
            out[f'prob_{name}'] = float(row[f'prob_{name}'])
            out[f'pred_{name}'] = int(pred.get(name, 0))
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'saved -> {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
