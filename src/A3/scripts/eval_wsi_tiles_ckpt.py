from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.wsi_manifest import WSITileDataset  # noqa: E402
from bach_mil.data.label_map import WSI_LABELS_DEFAULT  # noqa: E402
from bach_mil.models.encoder import build_patch_model  # noqa: E402
from bach_mil.utils.device import get_torch_device  # noqa: E402
from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402
from bach_mil.utils.metrics import multiclass_metrics  # noqa: E402


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--label_names', nargs='+', default=WSI_LABELS_DEFAULT)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    df = pd.read_csv(args.manifest_csv)
    label_names = list(args.label_names)
    df = df[df.label_name.isin(label_names)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError('No rows left after filtering label_names. Check --label_names / manifest_csv.')

    tf = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ds = WSITileDataset(df, transform=tf, return_label=True, label_names=label_names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = build_patch_model(
        model_name=args.model_name,
        num_classes=len(label_names),
        pretrained=False,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
    )
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    ys, probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval-wsi-tiles'):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logit = model(x)
            prob = torch.softmax(logit, dim=1)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    metrics = multiclass_metrics(y_true, y_prob)
    y_pred = y_prob.argmax(axis=1)
    cm = _confusion_matrix(y_true, y_pred, num_classes=len(label_names))

    save_json(
        {
            'manifest_csv': str(args.manifest_csv),
            'ckpt': str(args.ckpt),
            'num_tiles': int(len(df)),
            'label_names': label_names,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
        },
        out_dir / 'metrics.json',
    )
    pd.DataFrame(cm, index=[f'true_{k}' for k in label_names], columns=[f'pred_{k}' for k in label_names]).to_csv(out_dir / 'confusion_matrix.csv')
    print('metrics:', metrics)
    print(f'saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
