from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.label_map import WSI_LABELS_DEFAULT  # noqa: E402
from bach_mil.data.wsi_manifest import WSITileDataset  # noqa: E402
from bach_mil.models.encoder import build_patch_model  # noqa: E402
from bach_mil.utils.device import get_torch_device  # noqa: E402
from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402
from bach_mil.utils.metrics import multiclass_metrics  # noqa: E402
from bach_mil.utils.seed import seed_everything  # noqa: E402


def _build_split(df: pd.DataFrame, *, split_mode: str, val_ratio: float, num_folds: int, fold: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if split_mode == 'random_tile':
        rng = np.random.RandomState(int(seed))
        perm = rng.permutation(len(df))
        val_n = int(round(float(val_ratio) * float(len(df))))
        val_n = int(max(1, min(val_n, len(df) - 1)))
        va_df = df.iloc[perm[:val_n]].reset_index(drop=True)
        tr_df = df.iloc[perm[val_n:]].reset_index(drop=True)
    else:
        slides = sorted(df.slide_id.unique().tolist())
        kf = KFold(n_splits=min(int(num_folds), len(slides)), shuffle=True, random_state=int(seed))
        tr_idx, va_idx = list(kf.split(slides))[int(fold)]
        tr_slides = {slides[i] for i in tr_idx}
        va_slides = {slides[i] for i in va_idx}
        tr_df = df[df.slide_id.isin(tr_slides)].reset_index(drop=True)
        va_df = df[df.slide_id.isin(va_slides)].reset_index(drop=True)
    tr_slides = sorted(set(tr_df.slide_id.astype(str).tolist()))
    va_slides = sorted(set(va_df.slide_id.astype(str).tolist()))
    return tr_df, va_df, tr_slides, va_slides


def _evaluate(model, loader, device) -> dict:
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            prob = torch.softmax(model(x), dim=1)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
    return multiclass_metrics(np.concatenate(ys), np.concatenate(probs))


def _target_modules(model: nn.Module, include_classifier: bool) -> list[tuple[str, nn.Module]]:
    targets: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if (not include_classifier) and name == 'classifier':
                continue
            if any(key in name for key in ['qkv', 'proj', 'fc1', 'fc2', 'classifier']):
                targets.append((name, module))
        if isinstance(module, nn.Conv2d) and 'patch_embed.proj' in name:
            targets.append((name, module))
    return targets


def _apply_structured_prune(model: nn.Module, amount: float, include_classifier: bool) -> list[dict]:
    report = []
    for name, module in _target_modules(model, include_classifier=include_classifier):
        prune.ln_structured(module, name='weight', amount=float(amount), n=2, dim=0)
        prune.remove(module, 'weight')
        zeros = int(torch.count_nonzero(module.weight == 0).item())
        total = int(module.weight.numel())
        report.append({'module': name, 'zeros': zeros, 'total': total, 'zero_ratio': float(zeros / max(total, 1))})
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', type=str, required=True)
    parser.add_argument('--init_ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=WSI_LABELS_DEFAULT)
    parser.add_argument('--split_mode', type=str, default='slide_kfold', choices=['slide_kfold', 'random_tile'])
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--model_name', type=str, default='vit_large_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='token')
    parser.add_argument('--backbone_init_values', type=float, default=1.0)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder_lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
    parser.add_argument('--prune_ratio', type=float, default=0.1)
    parser.add_argument('--include_classifier', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    df = pd.read_csv(args.manifest_csv)
    label_names = list(args.label_names)
    df = df[df.label_name.isin(label_names)].reset_index(drop=True)
    tr_df, va_df, tr_slides, va_slides = _build_split(
        df,
        split_mode=args.split_mode,
        val_ratio=args.val_ratio,
        num_folds=args.num_folds,
        fold=args.fold,
        seed=args.seed,
    )

    train_tf = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
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
    ckpt = torch.load(args.init_ckpt, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    prune_report = _apply_structured_prune(model, amount=float(args.prune_ratio), include_classifier=bool(args.include_classifier))
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
        for p in model.backbone.parameters():
            p.requires_grad = epoch > int(args.freeze_encoder_epochs)
        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics = _evaluate(model, val_loader, device)
        metrics['epoch'] = int(epoch)
        metrics['train_loss'] = float(np.mean(losses)) if losses else float('nan')
        history.append(metrics)
        if float(metrics['macro_f1']) > best:
            best = float(metrics['macro_f1'])
            torch.save(
                {
                    'model': model.state_dict(),
                    'metrics': metrics,
                    'classes': label_names,
                    'feature_dim': int(getattr(model, 'feature_dim', 0)),
                    'prune_ratio': float(args.prune_ratio),
                },
                out_dir / 'best_pruned.pt',
            )
        save_json({'history': history}, out_dir / 'metrics.json')

    save_json(
        {
            'manifest_csv': str(args.manifest_csv),
            'init_ckpt': str(args.init_ckpt),
            'prune_ratio': float(args.prune_ratio),
            'include_classifier': bool(args.include_classifier),
            'train_slides': sorted(tr_slides),
            'val_slides': sorted(va_slides),
            'best_macro_f1': float(best),
            'modules': prune_report,
        },
        out_dir / 'prune_report.json',
    )
    with open(out_dir / 'split_meta.json', 'w', encoding='utf-8') as f:
        json.dump({'split_mode': args.split_mode, 'train_slides': tr_slides, 'val_slides': va_slides}, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
