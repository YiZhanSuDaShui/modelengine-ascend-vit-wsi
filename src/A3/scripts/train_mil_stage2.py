from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.wsi_bag_dataset import FeatureBagDataset, collate_bags
from bach_mil.models.mil import ClassWiseGatedAttentionMIL
from bach_mil.utils.device import get_torch_device
from bach_mil.utils.io import ensure_dir, save_json
from bach_mil.utils.metrics import multilabel_metrics, search_thresholds
from bach_mil.utils.seed import seed_everything


def evaluate(model, loader, device):
    model.eval()
    ys, probs = [], []
    slide_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            feats = batch['features'].to(device)
            y = batch['label'].to(device)
            tile_probs = batch['tile_probs']
            if tile_probs is not None:
                tile_probs = tile_probs.to(device)
            out = model(feats, tile_probs=tile_probs)
            prob = torch.sigmoid(out['logits'])
            ys.append(y.cpu().numpy()[None, :])
            probs.append(prob.cpu().numpy()[None, :])
            slide_ids.append(str(batch.get('slide_id', 'slide')))
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    thresholds = search_thresholds(y_true, y_prob)
    metrics = multilabel_metrics(y_true, y_prob, thresholds=thresholds)
    return metrics, thresholds, y_true, y_prob, slide_ids


def _val_pred_df(slide_ids: list[str], label_names: list[str], y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    assert y_true.shape == y_prob.shape
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int64)
    rows = []
    for i, sid in enumerate(slide_ids):
        row = {'slide_id': str(sid)}
        for ci, name in enumerate(label_names):
            row[f'true_{name}'] = int(y_true[i, ci])
            row[f'prob_{name}'] = float(y_prob[i, ci])
            row[f'pred_{name}'] = int(y_pred[i, ci])
        active = [name for ci, name in enumerate(label_names) if int(y_pred[i, ci]) == 1]
        row['pred_labels'] = ';'.join(active)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=['Normal', 'Benign', 'InSitu', 'Invasive'])
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_instances', type=int, default=1024)
    parser.add_argument('--instance_sampling', type=str, default='hybrid', choices=['random', 'tile_prior', 'hybrid'])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--save_val_predictions', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    bag_df = pd.read_csv(args.bag_csv)
    kf = KFold(n_splits=min(args.num_folds, len(bag_df)), shuffle=True, random_state=args.seed)
    splits = list(kf.split(bag_df))
    tr_idx, va_idx = splits[args.fold]
    tr_df = bag_df.iloc[tr_idx].reset_index(drop=True)
    va_df = bag_df.iloc[va_idx].reset_index(drop=True)

    train_ds = FeatureBagDataset(
        tr_df,
        args.feature_dir,
        label_names=args.label_names,
        max_instances=args.max_instances,
        training=True,
        instance_sampling=args.instance_sampling,
    )
    val_ds = FeatureBagDataset(va_df, args.feature_dir, label_names=args.label_names, max_instances=None, training=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bags)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_bags)

    model = ClassWiseGatedAttentionMIL(feature_dim=args.feature_dim, num_classes=len(args.label_names), hidden_dim=args.hidden_dim, dropout=args.dropout)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f'train epoch {epoch}', leave=False):
            feats = batch['features'].to(device)
            y = batch['label'].to(device)
            tile_probs = batch['tile_probs']
            if tile_probs is not None:
                tile_probs = tile_probs.to(device)
            out = model(feats, tile_probs=tile_probs)
            loss = criterion(out['logits'], y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        metrics, thresholds, y_true, y_prob, slide_ids = evaluate(model, val_loader, device)
        metrics['epoch'] = epoch
        metrics['train_loss'] = float(np.mean(losses))
        history.append(metrics)
        print(metrics)
        score = metrics['macro_f1']
        if score > best:
            best = score
            torch.save({
                'model': model.state_dict(),
                'feature_dim': args.feature_dim,
                'label_names': args.label_names,
                'thresholds': thresholds.tolist(),
                'metrics': metrics,
            }, out_dir / 'best.pt')
            save_json({k: float(v) for k, v in zip(args.label_names, thresholds.tolist())}, out_dir / 'thresholds.json')
            if args.save_val_predictions:
                _val_pred_df(slide_ids, list(args.label_names), y_true, y_prob, thresholds).to_csv(out_dir / 'val_predictions_best.csv', index=False)
        save_json({'history': history}, out_dir / 'metrics.json')

    print(f'best macro_f1={best:.4f}')


if __name__ == '__main__':
    main()
