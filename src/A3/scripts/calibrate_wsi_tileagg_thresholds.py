from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import save_json  # noqa: E402
from bach_mil.utils.metrics import search_thresholds  # noqa: E402


def _align_probs(tile_probs: torch.Tensor, prob_classes: list[str], label_names: list[str]) -> torch.Tensor:
    if list(prob_classes) == list(label_names):
        return tile_probs
    idx = []
    for name in label_names:
        if name not in prob_classes:
            raise ValueError(f'cannot align tile_probs: missing class={name} in prob_classes={prob_classes}')
        idx.append(int(prob_classes.index(name)))
    return tile_probs[:, idx]


def _agg_probs(tile_probs: torch.Tensor, agg: str, topk: int) -> torch.Tensor:
    # tile_probs: [N, C]
    if agg == 'mean_prob':
        return tile_probs.mean(dim=0)
    if agg == 'max_prob':
        return tile_probs.max(dim=0).values
    if agg == 'topk_mean_prob':
        k = int(max(1, min(int(topk), int(tile_probs.shape[0]))))
        topk_vals = torch.topk(tile_probs, k=k, dim=0).values
        return topk_vals.mean(dim=0)
    raise ValueError(f'unknown agg={agg}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', required=True)
    parser.add_argument('--agg', type=str, default='topk_mean_prob', choices=['mean_prob', 'max_prob', 'topk_mean_prob'])
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--all_positive_threshold', type=float, default=0.5)
    parser.add_argument('--all_negative_threshold', type=float, default=1.0)
    args = parser.parse_args()

    bags = pd.read_csv(args.bag_csv)
    label_names = list(args.label_names)
    for name in label_names:
        col = f'label_{name}'
        if col not in bags.columns:
            raise SystemExit(f'bag_csv missing column {col}')

    feat_dir = Path(args.feature_dir)
    slide_to_pt = {p.stem: p for p in feat_dir.glob('*.pt')}

    slide_ids: list[str] = []
    y_prob: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    for _, row in bags.iterrows():
        sid = str(row['slide_id'])
        pt = slide_to_pt.get(sid)
        if pt is None:
            continue
        blob = torch.load(pt, map_location='cpu')
        tile_probs = blob.get('tile_probs')
        if tile_probs is None:
            raise RuntimeError(f'{pt} missing key tile_probs')
        tile_probs = tile_probs.float()
        prob_classes = blob.get('classes')
        if not prob_classes:
            raise RuntimeError(f'{pt} missing classes for tile_probs alignment')
        aligned = _align_probs(tile_probs, list(prob_classes), label_names)
        slide_prob = _agg_probs(aligned, agg=str(args.agg), topk=int(args.topk)).cpu().numpy()
        slide_ids.append(sid)
        y_prob.append(slide_prob)
        y_true.append(np.asarray([int(row[f'label_{n}']) for n in label_names], dtype=np.int64))

    if not slide_ids:
        raise SystemExit('no overlapping slide_ids between bag_csv and feature_dir')

    y_prob_np = np.stack(y_prob, axis=0).astype(np.float32)
    y_true_np = np.stack(y_true, axis=0).astype(np.int64)

    thr = search_thresholds(y_true_np, y_prob_np)

    # Be conservative when validation labels are degenerate.
    for i, name in enumerate(label_names):
        s = int(y_true_np[:, i].sum())
        if s == 0:
            thr[i] = float(args.all_negative_threshold)
        elif s == int(len(y_true_np)):
            thr[i] = float(args.all_positive_threshold)

    out = {name: float(thr[i]) for i, name in enumerate(label_names)}
    save_json(out, args.out_json)
    print(f'saved -> {args.out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

