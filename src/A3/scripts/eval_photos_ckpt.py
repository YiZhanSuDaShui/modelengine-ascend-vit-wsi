from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.label_map import PHOTO_LABELS
from bach_mil.utils.metrics import multiclass_metrics
from bach_mil.runtime.patch_backends import create_patch_backend, softmax_numpy
from bach_mil.utils.photo_agg import PatchAggConfig, infer_image_patch_agg_backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--mode', type=str, choices=['resize', 'patch_agg'], default='patch_agg')
    parser.add_argument('--patch_crop_sizes', type=int, nargs='+', default=[512])
    parser.add_argument('--patch_stride', type=int, default=256)
    parser.add_argument('--patch_topk_per_size', type=int, default=8)
    parser.add_argument('--patch_min_tissue', type=float, default=0.4)
    parser.add_argument('--patch_working_max_side', type=int, default=1024)
    parser.add_argument('--patch_batch_size', type=int, default=64)
    parser.add_argument('--patch_agg', type=str, default='topk_mean_logit')
    parser.add_argument('--patch_logit_topk', type=int, default=8)
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'onnx', 'om', 'auto'])
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = create_patch_backend(
        backend=args.backend,
        ckpt_path=args.ckpt,
        model_name=args.model_name,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
        default_classes=PHOTO_LABELS,
        input_size=args.input_size,
        onnx_path=args.onnx_path,
        om_path=args.om_path,
        meta_json=args.meta_json,
        device_id=args.device_id,
    )

    df = pd.read_csv(args.split_csv)
    va_df = df[df.fold == args.fold].reset_index(drop=True)

    ys: list[int] = []
    probs: list[np.ndarray] = []
    rows = []
    try:
        for _, row in tqdm(va_df.iterrows(), total=len(va_df), desc='eval-photos'):
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
            r = {
                'image_id': str(row.image_id),
                'image_path': str(row.image_path),
                'true_label': PHOTO_LABELS[int(row.label)],
                'pred_label': PHOTO_LABELS[pred],
                'correct': int(pred == int(row.label)),
                'num_crops': num_crops,
            }
            for i, name in enumerate(PHOTO_LABELS):
                r[f'prob_{name}'] = float(prob[i])
            rows.append(r)
    finally:
        backend.close()

    y_true = np.asarray(ys, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float32)
    metrics = multiclass_metrics(y_true, y_prob)
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out_dir / 'val_predictions.csv', index=False)
    print(metrics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
