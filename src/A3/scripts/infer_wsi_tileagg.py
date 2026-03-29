from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import load_json  # noqa: E402


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
    if tile_probs.ndim != 2:
        raise ValueError(f'tile_probs must be [N,C], got shape={tuple(tile_probs.shape)}')
    if tile_probs.shape[0] == 0:
        raise ValueError('tile_probs is empty')
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
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=None)
    parser.add_argument('--agg', type=str, default='topk_mean_prob', choices=['mean_prob', 'max_prob', 'topk_mean_prob'])
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--thresholds_json', type=str, default=None)
    parser.add_argument('--default_threshold', type=float, default=0.5)
    parser.add_argument('--normal_fallback', action='store_true')
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    pt_paths = sorted(feature_dir.glob('*.pt'))
    if not pt_paths:
        raise SystemExit(f'no .pt files found in {feature_dir}')

    thresholds = None
    if args.thresholds_json is not None:
        thresholds = load_json(args.thresholds_json)

    rows = []
    for pt_path in tqdm(pt_paths, desc='infer-wsi-tileagg'):
        blob = torch.load(pt_path, map_location='cpu')
        tile_probs = blob.get('tile_probs')
        if tile_probs is None:
            raise RuntimeError(f'{pt_path} missing key tile_probs')
        tile_probs = tile_probs.float()
        prob_classes = blob.get('classes')
        if not prob_classes:
            raise RuntimeError(f'{pt_path} missing classes for tile_probs alignment')

        label_names = list(args.label_names) if args.label_names is not None else list(prob_classes)
        aligned = _align_probs(tile_probs, list(prob_classes), label_names)
        slide_prob = _agg_probs(aligned, agg=str(args.agg), topk=int(args.topk)).cpu().numpy()

        pred_flags = []
        active = []
        for i, name in enumerate(label_names):
            thr = float(args.default_threshold)
            if thresholds is not None and name in thresholds:
                thr = float(thresholds[name])
            flag = int(float(slide_prob[i]) >= thr)
            pred_flags.append(flag)
            if flag == 1:
                active.append(name)

        if args.normal_fallback and ('Normal' in label_names) and len(active) == 0:
            active = ['Normal']
            pred_flags[label_names.index('Normal')] = 1

        row = {
            'slide_id': str(pt_path.stem),
            'pred_labels': ';'.join(active),
            'n_tiles': int(aligned.shape[0]),
        }
        for i, name in enumerate(label_names):
            row[f'prob_{name}'] = float(slide_prob[i])
            row[f'pred_{name}'] = int(pred_flags[i])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'saved -> {args.out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

