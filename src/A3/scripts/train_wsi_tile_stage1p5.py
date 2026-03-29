from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
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
from bach_mil.utils.seed import seed_everything  # noqa: E402


def _pred_df(slide_ids: list[str], xs: list[int], ys: list[int], y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    y_pred = y_prob.argmax(axis=1)
    rows = []
    for i in range(len(y_true)):
        row = {
            'slide_id': slide_ids[i],
            'x': int(xs[i]),
            'y': int(ys[i]),
            'true_label': label_names[int(y_true[i])],
            'pred_label': label_names[int(y_pred[i])],
            'correct': int(y_true[i] == y_pred[i]),
        }
        for c, name in enumerate(label_names):
            row[f'prob_{name}'] = float(y_prob[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(model, loader, device, label_names: list[str]) -> tuple[dict, pd.DataFrame]:
    model.eval()
    ys, probs = [], []
    slide_ids: list[str] = []
    xs: list[int] = []
    ys0: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logit = model(x)
            prob = torch.softmax(logit, dim=1)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
            slide_ids.extend([str(s) for s in batch.get('slide_id', [])])
            xs.extend([int(v) for v in batch.get('x', [])])
            ys0.extend([int(v) for v in batch.get('y', [])])
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    metrics = multiclass_metrics(y_true, y_prob)
    pred_df = _pred_df(slide_ids, xs, ys0, y_true, y_prob, label_names=label_names)
    return metrics, pred_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=WSI_LABELS_DEFAULT)
    parser.add_argument('--split_mode', type=str, default='slide_kfold', choices=['slide_kfold', 'random_tile'])
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
    parser.add_argument('--init_ckpt', type=str, default=None)
    parser.add_argument('--save_val_predictions', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    df = pd.read_csv(args.manifest_csv)
    label_names = list(args.label_names)
    df = df[df.label_name.isin(label_names)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError('manifest_csv is empty after label filter.')

    if args.split_mode == 'random_tile':
        rng = np.random.RandomState(int(args.seed))
        perm = rng.permutation(len(df))
        val_n = int(round(float(args.val_ratio) * float(len(df))))
        val_n = int(max(1, min(val_n, len(df) - 1)))
        va_df = df.iloc[perm[:val_n]].reset_index(drop=True)
        tr_df = df.iloc[perm[val_n:]].reset_index(drop=True)
        tr_slides = sorted(set(tr_df.slide_id.astype(str).tolist()))
        va_slides = sorted(set(va_df.slide_id.astype(str).tolist()))
    else:
        slides = sorted(df.slide_id.unique().tolist())
        kf = KFold(n_splits=min(int(args.num_folds), len(slides)), shuffle=True, random_state=int(args.seed))
        splits = list(kf.split(slides))
        tr_idx, va_idx = splits[int(args.fold)]
        tr_slides = {slides[i] for i in tr_idx}
        va_slides = {slides[i] for i in va_idx}
        tr_df = df[df.slide_id.isin(tr_slides)].reset_index(drop=True)
        va_df = df[df.slide_id.isin(va_slides)].reset_index(drop=True)

    train_tf = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_ds = WSITileDataset(tr_df, transform=train_tf, return_label=True, label_names=label_names)
    val_ds = WSITileDataset(va_df, transform=val_tf, return_label=True, label_names=label_names)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_patch_model(
        model_name=args.model_name,
        num_classes=len(label_names),
        pretrained=False,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
    )
    if args.init_ckpt is not None and Path(args.init_ckpt).exists():
        ckpt = torch.load(args.init_ckpt, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.to(device)

    enc_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {'params': enc_params, 'lr': float(args.encoder_lr)},
            {'params': head_params, 'lr': float(args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    criterion = nn.CrossEntropyLoss()

    best = -1.0
    history: list[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        if epoch <= int(args.freeze_encoder_epochs):
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logit = model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        metrics, pred_df = evaluate(model, val_loader, device, label_names=label_names)
        metrics['epoch'] = int(epoch)
        metrics['train_loss'] = float(np.mean(losses)) if losses else float('nan')
        history.append(metrics)
        print(metrics)

        score = float(metrics['macro_f1'])
        if score > best:
            best = score
            torch.save(
                {'model': model.state_dict(), 'metrics': metrics, 'classes': label_names, 'feature_dim': int(getattr(model, 'feature_dim', 0))},
                out_dir / 'best.pt',
            )
            if args.save_val_predictions:
                pred_df.to_csv(out_dir / 'val_predictions_best.csv', index=False)
        save_json({'history': history}, out_dir / 'metrics.json')

    meta = {
        'manifest_csv': str(args.manifest_csv),
        'split_mode': str(args.split_mode),
        'val_ratio': float(args.val_ratio),
        'train_slides': sorted(tr_slides),
        'val_slides': sorted(va_slides),
        'label_names': label_names,
        'best_macro_f1': float(best),
    }
    with open(out_dir / 'split_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'best macro_f1={best:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
