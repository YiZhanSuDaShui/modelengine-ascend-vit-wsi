from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import json


def _read_best_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    history = d.get('history', [])
    best = max(history, key=lambda r: float(r.get('macro_f1', -1)))
    return {k: best.get(k) for k in ['epoch', 'acc', 'macro_f1', 'ovr_auc', 'train_loss']}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--init_backbone_weights', type=str, default=None)
    parser.add_argument('--init_backbone_safetensors', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--tissue_crop', action='store_true')
    parser.add_argument('--eval_mode', type=str, choices=['resize', 'patch_agg'], default='patch_agg')
    parser.add_argument('--eval_patch_crop_sizes', type=int, nargs='+', default=[512])
    parser.add_argument('--eval_patch_stride', type=int, default=256)
    parser.add_argument('--eval_patch_topk_per_size', type=int, default=8)
    parser.add_argument('--eval_patch_min_tissue', type=float, default=0.4)
    parser.add_argument('--eval_patch_working_max_side', type=int, default=1024)
    parser.add_argument('--eval_patch_batch_size', type=int, default=64)
    parser.add_argument('--eval_patch_agg', type=str, default='topk_mean_logit')
    parser.add_argument('--eval_patch_logit_topk', type=int, default=8)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).resolve().parent / 'train_patch_stage1.py'
    rows = []
    for fold in range(args.n_folds):
        run_dir = out_root / f'fold{fold}'
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            '--data_root', args.data_root,
            '--split_csv', args.split_csv,
            '--fold', str(fold),
            '--out_dir', str(run_dir),
            '--model_name', args.model_name,
            '--backbone_pool', args.backbone_pool,
            '--input_size', str(args.input_size),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
            '--lr', str(args.lr),
            '--encoder_lr', str(args.encoder_lr),
            '--weight_decay', str(args.weight_decay),
            '--freeze_encoder_epochs', str(args.freeze_encoder_epochs),
            '--seed', str(args.seed),
            '--eval_mode', args.eval_mode,
            '--eval_patch_stride', str(args.eval_patch_stride),
            '--eval_patch_topk_per_size', str(args.eval_patch_topk_per_size),
            '--eval_patch_min_tissue', str(args.eval_patch_min_tissue),
            '--eval_patch_working_max_side', str(args.eval_patch_working_max_side),
            '--eval_patch_batch_size', str(args.eval_patch_batch_size),
            '--eval_patch_agg', str(args.eval_patch_agg),
            '--eval_patch_logit_topk', str(args.eval_patch_logit_topk),
            '--eval_patch_crop_sizes',
            *[str(int(cs)) for cs in args.eval_patch_crop_sizes],
            '--save_val_predictions',
        ]

        if args.init_backbone_weights:
            cmd += ['--init_backbone_weights', args.init_backbone_weights]
        if args.init_backbone_safetensors:
            cmd += ['--init_backbone_safetensors', args.init_backbone_safetensors]
        if args.backbone_init_values is not None:
            cmd += ['--backbone_init_values', str(args.backbone_init_values)]
        if args.pretrained:
            cmd += ['--pretrained']
        if args.tissue_crop:
            cmd += ['--tissue_crop']

        log_path = run_dir / 'train.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(cmd) + '\n')
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'Fold {fold} failed. See log: {log_path}')

        best = _read_best_metrics(run_dir)
        row = {'fold': fold, **{k: float(best[k]) for k in ['epoch', 'acc', 'macro_f1', 'ovr_auc', 'train_loss']}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / 'cv_summary_partial.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'cv_summary.csv', index=False)
    agg = df[['acc', 'macro_f1', 'ovr_auc']].agg(['mean', 'std']).reset_index()
    agg.to_csv(out_root / 'cv_summary_mean_std.csv', index=False)
    print(df)
    print(agg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
