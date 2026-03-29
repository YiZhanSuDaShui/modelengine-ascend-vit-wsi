from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.photos import PhotoDataset
from bach_mil.data.wsi_manifest import WSITileDataset
from bach_mil.data.label_map import PHOTO_LABELS, WSI_LABELS_DEFAULT
from bach_mil.models.encoder import build_patch_model
from bach_mil.utils.device import get_torch_device
from bach_mil.utils.io import ensure_dir, save_json
from bach_mil.utils.metrics import multiclass_metrics
from bach_mil.utils.seed import seed_everything
from bach_mil.utils.photo_agg import PatchAggConfig, infer_image_patch_agg


class _TileDatasetAdapter(Dataset):
    """Adapt WSITileDataset sample keys to match PhotoDataset for concat training."""

    def __init__(self, tile_ds: Dataset) -> None:
        self.tile_ds = tile_ds

    def __len__(self):
        return len(self.tile_ds)

    def __getitem__(self, index):
        s = self.tile_ds[index]
        slide_id = s.get('slide_id', 'slide')
        x = s.get('x', 0)
        y = s.get('y', 0)
        return {
            'image': s['image'],
            'label': s['label'],
            'image_id': f'{slide_id}_{x}_{y}',
            'image_path': f'{slide_id}:{x}:{y}',
        }


def _pred_df_from_probs(image_ids: list[str], image_paths: list[str], y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    y_pred = y_prob.argmax(axis=1)
    rows = []
    for i in range(len(image_ids)):
        row = {
            'image_id': image_ids[i],
            'image_path': image_paths[i],
            'true_label': PHOTO_LABELS[int(y_true[i])],
            'pred_label': PHOTO_LABELS[int(y_pred[i])],
            'correct': int(y_true[i] == y_pred[i]),
        }
        for c, name in enumerate(PHOTO_LABELS):
            row[f'prob_{name}'] = float(y_prob[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def eval_model_resize(model, loader, device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    ys, probs = [], []
    image_ids: list[str] = []
    image_paths: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='valid', leave=False):
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logit = model(x)
            prob = torch.softmax(logit, dim=1)
            ys.append(y.cpu().numpy())
            probs.append(prob.cpu().numpy())
            image_ids.extend(list(batch.get('image_id', [])))
            image_paths.extend(list(batch.get('image_path', [])))
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    metrics = multiclass_metrics(y_true, y_prob)
    pred_df = _pred_df_from_probs(image_ids, image_paths, y_true, y_prob)
    return metrics, pred_df


def eval_model_patch_agg(model, df: pd.DataFrame, device: str, cfg: PatchAggConfig, batch_size: int) -> tuple[dict, pd.DataFrame]:
    model.eval()
    ys = []
    probs = []
    image_ids: list[str] = []
    image_paths: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='valid-patch-agg', leave=False):
        img = Image.open(row.image_path).convert('RGB')
        prob, _meta = infer_image_patch_agg(
            model=model,
            img=img,
            device=device,
            num_classes=len(PHOTO_LABELS),
            cfg=cfg,
            batch_size=batch_size,
        )
        ys.append(int(row.label))
        probs.append(prob)
        image_ids.append(str(row.image_id))
        image_paths.append(str(row.image_path))
    y_true = np.asarray(ys, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float32)
    metrics = multiclass_metrics(y_true, y_prob)
    pred_df = _pred_df_from_probs(image_ids, image_paths, y_true, y_prob)
    return metrics, pred_df


def _load_state_dict_from_path(path: Path):
    suffix = path.suffix.lower()
    if suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                'Failed to import safetensors. Please install it (pip install safetensors) '
                'or do not pass a .safetensors file.'
            ) from e
        return load_file(str(path), device='cpu')

    blob = torch.load(path, map_location='cpu')
    if isinstance(blob, OrderedDict):
        return blob
    if isinstance(blob, dict):
        for key in ['state_dict', 'model', 'teacher', 'backbone']:
            sd = blob.get(key)
            if isinstance(sd, (dict, OrderedDict)):
                return sd
    raise RuntimeError(f'Unsupported backbone checkpoint format: {path}')


def _load_backbone_weights(model, weight_path: Path, out_dir: Path) -> None:
    sd = _load_state_dict_from_path(weight_path)
    incompatible = model.backbone.load_state_dict(sd, strict=False)
    save_json(
        {
            'init_backbone_path': str(weight_path),
            'missing_keys': list(getattr(incompatible, 'missing_keys', [])),
            'unexpected_keys': list(getattr(incompatible, 'unexpected_keys', [])),
            'num_tensors': int(len(sd)),
        },
        out_dir / 'init_backbone_load_report.json',
    )
    print(
        'Loaded backbone weights:',
        f"path={weight_path}",
        f"num_tensors={len(sd)}",
        f"missing={len(getattr(incompatible, 'missing_keys', []))}",
        f"unexpected={len(getattr(incompatible, 'unexpected_keys', []))}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--extra_tiles_csv', type=str, default=None)
    parser.add_argument('--init_ckpt', type=str, default=None)
    parser.add_argument('--init_backbone_weights', type=str, default=None)
    parser.add_argument('--init_backbone_safetensors', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--tissue_crop', action='store_true')
    parser.add_argument('--eval_mode', type=str, choices=['resize', 'patch_agg'], default='resize')
    parser.add_argument('--eval_patch_crop_sizes', type=int, nargs='+', default=[512])
    parser.add_argument('--eval_patch_stride', type=int, default=256)
    parser.add_argument('--eval_patch_topk_per_size', type=int, default=8)
    parser.add_argument('--eval_patch_min_tissue', type=float, default=0.4)
    parser.add_argument('--eval_patch_working_max_side', type=int, default=1024)
    parser.add_argument('--eval_patch_batch_size', type=int, default=64)
    parser.add_argument('--eval_patch_agg', type=str, default='topk_mean_logit')
    parser.add_argument('--eval_patch_logit_topk', type=int, default=8)
    parser.add_argument('--save_val_predictions', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    df = pd.read_csv(args.split_csv)
    tr_df = df[df.fold != args.fold].reset_index(drop=True)
    va_df = df[df.fold == args.fold].reset_index(drop=True)

    train_ds = PhotoDataset(tr_df, input_size=args.input_size, training=True, tissue_crop=args.tissue_crop)
    val_ds = PhotoDataset(va_df, input_size=args.input_size, training=False)

    if args.extra_tiles_csv is not None and Path(args.extra_tiles_csv).exists():
        extra_df = pd.read_csv(args.extra_tiles_csv)
        tile_tf = T.Compose([
            T.Resize((args.input_size, args.input_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        extra_ds = WSITileDataset(extra_df[extra_df.label_name.isin(WSI_LABELS_DEFAULT)], transform=tile_tf, return_label=True, label_names=WSI_LABELS_DEFAULT)
        train_ds = ConcatDataset([train_ds, _TileDatasetAdapter(extra_ds)])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Keep offline-friendly behavior:
    # - only download pretrained weights if user explicitly requests it via --pretrained
    # - if local backbone weights are provided, prefer them and avoid any online download.
    init_backbone_path = None
    if args.init_backbone_weights is not None:
        init_backbone_path = Path(args.init_backbone_weights)
    elif args.init_backbone_safetensors is not None:
        init_backbone_path = Path(args.init_backbone_safetensors)

    pretrained = bool(args.pretrained) and (init_backbone_path is None)
    model = build_patch_model(
        model_name=args.model_name,
        num_classes=len(PHOTO_LABELS),
        pretrained=pretrained,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
    )

    if init_backbone_path is not None and init_backbone_path.exists():
        _load_backbone_weights(model=model, weight_path=init_backbone_path, out_dir=out_dir)

    if args.init_ckpt is not None and Path(args.init_ckpt).exists():
        ckpt = torch.load(args.init_ckpt, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.to(device)

    enc_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': args.encoder_lr},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch <= args.freeze_encoder_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True

        losses = []
        pbar = tqdm(train_loader, desc=f'train epoch {epoch}', leave=False)
        for batch in pbar:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            logit = model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f'{np.mean(losses):.4f}')
        scheduler.step()

        if args.eval_mode == 'patch_agg':
            cfg = PatchAggConfig(
                input_size=args.input_size,
                crop_sizes=tuple(int(x) for x in args.eval_patch_crop_sizes),
                stride=int(args.eval_patch_stride),
                topk_per_size=int(args.eval_patch_topk_per_size),
                min_tissue=float(args.eval_patch_min_tissue),
                working_max_side=int(args.eval_patch_working_max_side),
                agg=str(args.eval_patch_agg),
                logit_topk=int(args.eval_patch_logit_topk),
            )
            metrics, pred_df = eval_model_patch_agg(model, va_df, device, cfg, batch_size=args.eval_patch_batch_size)
        else:
            metrics, pred_df = eval_model_resize(model, val_loader, device)
        metrics['train_loss'] = float(np.mean(losses))
        metrics['epoch'] = epoch
        history.append(metrics)
        print(metrics)
        score = metrics['macro_f1']
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'metrics': metrics, 'feature_dim': model.feature_dim, 'classes': PHOTO_LABELS}, out_dir / 'best.pt')
            if args.save_val_predictions:
                pred_df.to_csv(out_dir / 'val_predictions_best.csv', index=False)
        save_json({'history': history}, out_dir / 'metrics.json')

    print(f'best macro_f1={best:.4f}')


if __name__ == '__main__':
    main()
