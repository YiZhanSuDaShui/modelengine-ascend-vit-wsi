from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csvs', nargs='+', required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--id_col', type=str, default='slide_id')
    parser.add_argument('--prob_prefix', type=str, default='prob_')
    parser.add_argument('--weights', nargs='*', type=float, default=None)
    args = parser.parse_args()

    in_csvs = [Path(p) for p in args.in_csvs]
    dfs = [pd.read_csv(p) for p in in_csvs]
    for p, df in zip(in_csvs, dfs):
        if args.id_col not in df.columns:
            raise SystemExit(f'{p} missing id_col={args.id_col}')

    prob_cols = None
    for df in dfs:
        cols = sorted([c for c in df.columns if c.startswith(args.prob_prefix)])
        prob_cols = set(cols) if prob_cols is None else (prob_cols & set(cols))
    prob_cols = sorted(list(prob_cols or []))
    if not prob_cols:
        raise SystemExit(f'no prob columns found with prefix={args.prob_prefix}')

    weights = args.weights
    if weights is None or len(weights) == 0:
        weights = [1.0] * len(dfs)
    if len(weights) != len(dfs):
        raise SystemExit('len(weights) must match len(in_csvs)')
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    out = None
    for wi, df in zip(w.tolist(), dfs):
        sub = df[[args.id_col] + prob_cols].copy()
        sub = sub.set_index(args.id_col)
        sub = sub.astype(np.float64) * float(wi)
        out = sub if out is None else out.add(sub, fill_value=0.0)
    out = out.reset_index()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f'saved -> {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

