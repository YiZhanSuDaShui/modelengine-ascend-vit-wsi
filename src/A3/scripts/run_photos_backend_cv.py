from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.label_map import PHOTO_LABELS  # noqa: E402
from bach_mil.runtime.patch_backends import create_patch_backend, softmax_numpy  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402
from bach_mil.utils.metrics import multiclass_metrics  # noqa: E402
from bach_mil.utils.photo_agg import PatchAggConfig, infer_image_patch_agg_backend  # noqa: E402


def _resolve_fold_path(pattern: str | None, fold: int) -> str | None:
    if pattern is None:
        return None
    return str(pattern).format(fold=int(fold))


def _build_backend_for_fold(args, defaults: dict, artifacts: dict, fold: int):
    ckpt_path = args.ckpt
    onnx_path = args.onnx_path
    om_path = args.om_path
    meta_json = args.meta_json
    if args.strict_fold_ckpt:
        ckpt_path = _resolve_fold_path(args.fold_ckpt_pattern, fold) or ckpt_path
        onnx_path = _resolve_fold_path(args.fold_onnx_pattern, fold) or onnx_path
        om_path = _resolve_fold_path(args.fold_om_pattern, fold) or om_path
        meta_json = _resolve_fold_path(args.fold_meta_pattern, fold) or meta_json
    backend = create_patch_backend(
        backend=args.backend,
        ckpt_path=ckpt_path,
        model_name=args.model_name,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
        default_classes=PHOTO_LABELS,
        input_size=args.input_size,
        onnx_path=onnx_path,
        om_path=om_path,
        meta_json=meta_json,
        device_id=args.device_id,
    )
    refs = {
        'ckpt': str(ckpt_path) if ckpt_path is not None else None,
        'onnx_path': str(onnx_path) if onnx_path is not None else None,
        'om_path': str(om_path) if om_path is not None else None,
        'meta_json': str(meta_json) if meta_json is not None else None,
    }
    return backend, refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_csv', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'onnx', 'om', 'auto'])
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--backbone_pool', type=str, default=None)
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--mode', type=str, default=None, choices=['resize', 'patch_agg'])
    parser.add_argument('--patch_crop_sizes', type=int, nargs='+', default=None)
    parser.add_argument('--patch_stride', type=int, default=None)
    parser.add_argument('--patch_topk_per_size', type=int, default=None)
    parser.add_argument('--patch_min_tissue', type=float, default=None)
    parser.add_argument('--patch_working_max_side', type=int, default=None)
    parser.add_argument('--patch_batch_size', type=int, default=None)
    parser.add_argument('--patch_agg', type=str, default=None)
    parser.add_argument('--patch_logit_topk', type=int, default=None)
    parser.add_argument('--strict_fold_ckpt', action='store_true')
    parser.add_argument('--fold_ckpt_pattern', type=str, default=None)
    parser.add_argument('--fold_onnx_pattern', type=str, default=None)
    parser.add_argument('--fold_om_pattern', type=str, default=None)
    parser.add_argument('--fold_meta_pattern', type=str, default=None)
    args = parser.parse_args()

    defaults = get_task_defaults('photos')
    artifacts = get_default_backend_artifacts('photos')
    split_csv = Path(args.split_csv) if args.split_csv is not None else Path(defaults['split_csv'])
    out_dir = ensure_dir(args.out_dir)
    args.ckpt = args.ckpt or str(defaults['ckpt'])
    args.onnx_path = args.onnx_path or str(artifacts['onnx'])
    args.om_path = args.om_path or str(artifacts['om'])
    if args.meta_json is None:
        default_meta_json = artifacts['meta_json']
        if args.backend == 'om' or (args.backend == 'auto' and Path(args.om_path).exists()):
            default_meta_json = artifacts.get('om_meta_json', default_meta_json)
        args.meta_json = str(default_meta_json)
    args.model_name = args.model_name or defaults['model_name']
    args.backbone_pool = args.backbone_pool or defaults['backbone_pool']
    if args.backbone_init_values is None:
        args.backbone_init_values = defaults['backbone_init_values']
    if args.input_size is None:
        args.input_size = defaults['input_size']
    args.mode = args.mode or defaults['mode']
    args.patch_crop_sizes = args.patch_crop_sizes or defaults['patch_crop_sizes']
    args.patch_stride = defaults['patch_stride'] if args.patch_stride is None else args.patch_stride
    args.patch_topk_per_size = defaults['patch_topk_per_size'] if args.patch_topk_per_size is None else args.patch_topk_per_size
    args.patch_min_tissue = defaults['patch_min_tissue'] if args.patch_min_tissue is None else args.patch_min_tissue
    args.patch_working_max_side = defaults['patch_working_max_side'] if args.patch_working_max_side is None else args.patch_working_max_side
    args.patch_batch_size = defaults['batch_size'] if args.patch_batch_size is None else args.patch_batch_size
    args.patch_agg = args.patch_agg or defaults['patch_agg']
    args.patch_logit_topk = defaults['patch_logit_topk'] if args.patch_logit_topk is None else args.patch_logit_topk
    if args.fold_ckpt_pattern is None and args.strict_fold_ckpt:
        args.fold_ckpt_pattern = str(Path(args.ckpt).resolve().parents[1] / 'fold{fold}' / 'best.pt')

    df = pd.read_csv(split_csv)
    folds = sorted(int(x) for x in df['fold'].unique())
    fold_rows = []
    for fold in folds:
        fold_dir = ensure_dir(out_dir / f'fold{fold}')
        va_df = df[df.fold == fold].reset_index(drop=True)
        backend, refs = _build_backend_for_fold(args, defaults, artifacts, fold)
        ys = []
        probs = []
        pred_rows = []
        try:
            for _, row in tqdm(va_df.iterrows(), total=len(va_df), desc=f'photos-cv-fold{fold}'):
                img = Image.open(row.image_path).convert('RGB')
                if args.mode == 'patch_agg':
                    cfg = PatchAggConfig(
                        input_size=args.input_size,
                        crop_sizes=tuple(int(x) for x in args.patch_crop_sizes),
                        stride=int(args.patch_stride),
                        topk_per_size=int(args.patch_topk_per_size),
                        min_tissue=float(args.patch_min_tissue),
                        working_max_side=int(args.patch_working_max_side),
                        agg=str(args.patch_agg),
                        logit_topk=int(args.patch_logit_topk),
                    )
                    prob, meta = infer_image_patch_agg_backend(
                        backend=backend,
                        img=img,
                        num_classes=len(PHOTO_LABELS),
                        cfg=cfg,
                        batch_size=args.patch_batch_size,
                    )
                    num_crops = int(meta.get('num_crops', 0))
                else:
                    from torchvision import transforms as T
                    tf = T.Compose([
                        T.Resize((args.input_size, args.input_size)),
                        T.ToTensor(),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
                    x = tf(img).unsqueeze(0)
                    prob = softmax_numpy(backend.predict_batch(x).logits)[0]
                    num_crops = 1
                ys.append(int(row.label))
                probs.append(prob)
                pred = int(prob.argmax())
                rec = {
                    'image_id': str(row.image_id),
                    'image_path': str(row.image_path),
                    'true_label': PHOTO_LABELS[int(row.label)],
                    'pred_label': PHOTO_LABELS[pred],
                    'correct': int(pred == int(row.label)),
                    'num_crops': num_crops,
                    'backend_actual': str(getattr(backend, 'backend_name', args.backend)),
                    'ckpt': refs['ckpt'],
                    'onnx_path': refs['onnx_path'],
                    'om_path': refs['om_path'],
                    'meta_json': refs['meta_json'],
                }
                for i, name in enumerate(PHOTO_LABELS):
                    rec[f'prob_{name}'] = float(prob[i])
                pred_rows.append(rec)
        finally:
            backend.close()
        y_true = np.asarray(ys, dtype=np.int64)
        y_prob = np.asarray(probs, dtype=np.float32)
        metrics = multiclass_metrics(y_true, y_prob)
        metrics['fold'] = int(fold)
        metrics['backend_actual'] = str(pred_rows[0]['backend_actual']) if pred_rows else str(args.backend)
        metrics['ckpt'] = refs['ckpt']
        metrics['onnx_path'] = refs['onnx_path']
        metrics['om_path'] = refs['om_path']
        metrics['meta_json'] = refs['meta_json']
        fold_rows.append(metrics)
        pd.DataFrame(pred_rows).to_csv(fold_dir / 'val_predictions.csv', index=False)
        save_json(metrics, fold_dir / 'metrics.json')

    fold_df = pd.DataFrame(fold_rows).sort_values('fold').reset_index(drop=True)
    fold_df.to_csv(out_dir / 'cv_summary.csv', index=False)
    metric_cols = [c for c in ['acc', 'macro_f1', 'ovr_auc'] if c in fold_df.columns]
    agg = fold_df[metric_cols].agg(['mean', 'std']).reset_index()
    agg.to_csv(out_dir / 'cv_summary_mean_std.csv', index=False)
    save_json(
        {
            'split_csv': str(split_csv),
            'backend': str(args.backend),
            'ckpt': str(args.ckpt),
            'onnx_path': str(args.onnx_path),
            'om_path': str(args.om_path),
            'meta_json': str(args.meta_json),
            'strict_fold_ckpt': bool(args.strict_fold_ckpt),
            'fold_ckpt_pattern': str(args.fold_ckpt_pattern) if args.fold_ckpt_pattern is not None else None,
            'fold_onnx_pattern': str(args.fold_onnx_pattern) if args.fold_onnx_pattern is not None else None,
            'fold_om_pattern': str(args.fold_om_pattern) if args.fold_om_pattern is not None else None,
            'fold_meta_pattern': str(args.fold_meta_pattern) if args.fold_meta_pattern is not None else None,
            'model_name': str(args.model_name),
            'mode': str(args.mode),
        },
        out_dir / 'config.json',
    )
    print(fold_df)
    print(agg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
