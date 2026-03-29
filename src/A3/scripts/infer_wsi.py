from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.models.mil import ClassWiseGatedAttentionMIL
from bach_mil.utils.device import get_torch_device
from bach_mil.utils.io import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--mil_ckpt', type=str, required=True)
    parser.add_argument('--thresholds_json', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--max_instances', type=int, default=None)
    parser.add_argument('--instance_select', type=str, choices=['head', 'tile_prior'], default='head')
    args = parser.parse_args()

    device = get_torch_device()
    ckpt = torch.load(args.mil_ckpt, map_location='cpu')
    label_names = ckpt.get('label_names', ['Normal', 'Benign', 'InSitu', 'Invasive'])
    model = ClassWiseGatedAttentionMIL(feature_dim=args.feature_dim, num_classes=len(label_names))
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    t_json = load_json(args.thresholds_json)
    thresholds = torch.tensor([t_json[k] for k in label_names], dtype=torch.float32)

    rows = []
    for pt_path in tqdm(sorted(Path(args.feature_dir).glob('*.pt')), desc='infer-wsi'):
        blob = torch.load(pt_path, map_location='cpu')
        feats = blob['features'].float()
        tile_probs = blob.get('tile_probs')
        if tile_probs is not None:
            tile_probs = tile_probs.float()
        # Align tile_probs columns to MIL label_names when possible.
        if tile_probs is not None and tile_probs.ndim == 2 and tile_probs.shape[0] == feats.shape[0] and tile_probs.shape[1] != len(label_names):
            prob_classes = blob.get('classes')
            if isinstance(prob_classes, (list, tuple)) and prob_classes:
                idx = []
                ok = True
                for name in label_names:
                    if name in prob_classes:
                        idx.append(int(prob_classes.index(name)))
                    else:
                        ok = False
                        break
                if ok and len(idx) == len(label_names):
                    tile_probs = tile_probs[:, idx]
            # Heuristic fallback: common case is 4-class probs with leading Normal.
            if tile_probs is not None and tile_probs.shape[1] == 4 and len(label_names) == 3 and 'Normal' not in label_names:
                tile_probs = tile_probs[:, 1:4]
        if args.max_instances is not None and len(feats) > args.max_instances:
            if args.instance_select == 'tile_prior' and tile_probs is not None and tile_probs.ndim == 2 and tile_probs.shape[1] >= 1:
                if 'Normal' in label_names and tile_probs.shape[1] >= 2:
                    score = tile_probs[:, 1:].max(dim=1).values
                else:
                    score = tile_probs.max(dim=1).values
                idx = torch.topk(score, k=args.max_instances, largest=True).indices
                feats = feats[idx]
                tile_probs = tile_probs[idx]
            else:
                feats = feats[:args.max_instances]
                if tile_probs is not None:
                    tile_probs = tile_probs[:args.max_instances]
        feats = feats.to(device)
        if tile_probs is not None:
            tile_probs = tile_probs.to(device)
        with torch.no_grad():
            out = model(feats, tile_probs=tile_probs)
            prob = torch.sigmoid(out['logits']).cpu()
        pred_multi = (prob >= thresholds).int().tolist()
        active = [name for flag, name in zip(pred_multi, label_names) if flag == 1]
        row = {'slide_id': pt_path.stem, 'pred_labels': ';'.join(active)}
        for i, name in enumerate(label_names):
            row[f'prob_{name}'] = float(prob[i])
            row[f'pred_{name}'] = int(pred_multi[i])
        if len(active) == 0 and 'Normal' in label_names:
            row['pred_labels'] = 'Normal'
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f'saved -> {args.out_csv}')


if __name__ == '__main__':
    main()
