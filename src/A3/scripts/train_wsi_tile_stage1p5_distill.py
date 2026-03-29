from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _evaluate(model, loader, device, label_names: list[str]) -> tuple[dict, pd.DataFrame]:
    model.eval()
    ys, probs = [], []
    slide_ids: list[str] = []
    xs: list[int] = []
    ys0: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            prob = torch.softmax(model(x), dim=1)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
            slide_ids.extend([str(s) for s in batch.get('slide_id', [])])
            xs.extend([int(v) for v in batch.get('x', [])])
            ys0.extend([int(v) for v in batch.get('y', [])])
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    rows = []
    y_pred = y_prob.argmax(axis=1)
    for i in range(len(y_true)):
        row = {
            'slide_id': slide_ids[i],
            'x': int(xs[i]),
            'y': int(ys0[i]),
            'true_label': label_names[int(y_true[i])],
            'pred_label': label_names[int(y_pred[i])],
            'correct': int(y_true[i] == y_pred[i]),
        }
        for c, name in enumerate(label_names):
            row[f'prob_{name}'] = float(y_prob[i, c])
        rows.append(row)
    return multiclass_metrics(y_true, y_prob), pd.DataFrame(rows)


def _load_ckpt(path: str | Path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt['model'] if 'model' in ckpt else ckpt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--teacher_ckpt', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=WSI_LABELS_DEFAULT)
    parser.add_argument('--split_mode', type=str, default='slide_kfold', choices=['slide_kfold', 'random_tile'])
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--teacher_model_name', type=str, default='vit_large_patch16_224')
    parser.add_argument('--teacher_backbone_pool', type=str, default='token')
    parser.add_argument('--teacher_backbone_init_values', type=float, default=1.0)
    parser.add_argument('--student_model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--student_backbone_pool', type=str, default='token')
    parser.add_argument('--student_backbone_init_values', type=float, default=1.0)
    parser.add_argument('--student_init_ckpt', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--kd_alpha', type=float, default=0.5)
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

    teacher = build_patch_model(
        model_name=args.teacher_model_name,
        num_classes=len(label_names),
        pretrained=False,
        backbone_pool=args.teacher_backbone_pool,
        backbone_init_values=args.teacher_backbone_init_values,
    )
    teacher.load_state_dict(_load_ckpt(args.teacher_ckpt), strict=False)
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_patch_model(
        model_name=args.student_model_name,
        num_classes=len(label_names),
        pretrained=False,
        backbone_pool=args.student_backbone_pool,
        backbone_init_values=args.student_backbone_init_values,
    )
    if args.student_init_ckpt is not None and Path(args.student_init_ckpt).exists():
        student.load_state_dict(_load_ckpt(args.student_init_ckpt), strict=False)
    student.to(device)

    enc_params = list(student.backbone.parameters())
    head_params = list(student.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {'params': enc_params, 'lr': float(args.encoder_lr)},
            {'params': head_params, 'lr': float(args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))
    ce_loss = nn.CrossEntropyLoss()

    best = -1.0
    history: list[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        for p in student.backbone.parameters():
            p.requires_grad = epoch > int(args.freeze_encoder_epochs)
        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss_ce = ce_loss(student_logits, y)
            student_log_probs = F.log_softmax(student_logits / float(args.temperature), dim=1)
            teacher_probs = F.softmax(teacher_logits / float(args.temperature), dim=1)
            loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * float(args.temperature) * float(args.temperature)
            loss = float(args.kd_alpha) * loss_ce + (1.0 - float(args.kd_alpha)) * loss_kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        metrics, pred_df = _evaluate(student, val_loader, device, label_names=label_names)
        metrics['epoch'] = int(epoch)
        metrics['train_loss'] = float(np.mean(losses)) if losses else float('nan')
        history.append(metrics)
        print(metrics)

        score = float(metrics['macro_f1'])
        if score > best:
            best = score
            torch.save(
                {
                    'model': student.state_dict(),
                    'metrics': metrics,
                    'classes': label_names,
                    'feature_dim': int(getattr(student, 'feature_dim', 0)),
                    'teacher_ckpt': str(args.teacher_ckpt),
                    'student_model_name': str(args.student_model_name),
                },
                out_dir / 'best_student.pt',
            )
            if args.save_val_predictions:
                pred_df.to_csv(out_dir / 'val_predictions_best_student.csv', index=False)
        save_json({'history': history}, out_dir / 'metrics.json')

    meta = {
        'manifest_csv': str(args.manifest_csv),
        'teacher_ckpt': str(args.teacher_ckpt),
        'split_mode': str(args.split_mode),
        'val_ratio': float(args.val_ratio),
        'train_slides': sorted(tr_slides),
        'val_slides': sorted(va_slides),
        'label_names': label_names,
        'best_macro_f1': float(best),
    }
    with open(out_dir / 'split_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
