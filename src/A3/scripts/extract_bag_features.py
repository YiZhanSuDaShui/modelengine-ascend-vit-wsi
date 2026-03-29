from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.data.wsi_manifest import WSITileDataset, list_wsi_paths, scan_slide_for_tiles
from bach_mil.runtime.patch_backends import create_patch_backend
from bach_mil.utils.io import ensure_dir


def save_grouped_features(
    df: pd.DataFrame,
    feats: torch.Tensor,
    probs: torch.Tensor,
    coords: torch.Tensor,
    out_dir: Path,
    *,
    classes: list[str],
    encoder_ckpt: str,
    model_name: str,
):
    start = 0
    for slide_id, sub_df in df.groupby('slide_id', sort=False):
        n = len(sub_df)
        if 'level' in sub_df:
            levels = sorted({int(x) for x in sub_df['level'].tolist()})
        else:
            levels = []
        if 'tile_size' in sub_df:
            tile_sizes = sorted({int(x) for x in sub_df['tile_size'].tolist()})
        else:
            tile_sizes = []
        blob = {
            'features': feats[start:start+n].cpu(),
            'tile_probs': probs[start:start+n].cpu(),
            'coords': coords[start:start+n].cpu(),
            'slide_id': slide_id,
            'classes': list(classes),
            'encoder_ckpt': str(encoder_ckpt),
            'model_name': str(model_name),
            'levels': levels,
            'tile_sizes': tile_sizes,
        }
        torch.save(blob, out_dir / f'{slide_id}.pt')
        start += n


def run_manifest(args, backend, classes):
    df = pd.read_csv(args.manifest_csv)
    tf = T.Compose([
        T.Resize((args.tile_size, args.tile_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ds = WSITileDataset(df, transform=tf, return_label=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    feats, probs, coords, rows = [], [], [], []
    for batch in tqdm(loader, desc='extract-manifest'):
        out = backend.predict_batch(batch['image'])
        feats.append(torch.from_numpy(out.features))
        probs.append(torch.from_numpy(out.probs))
        xy = torch.stack([batch['x'], batch['y']], dim=1)
        coords.append(xy)
        rows.append(pd.DataFrame({'slide_id': batch['slide_id'], 'level': batch['level'], 'tile_size': batch['tile_size']}))
    feats = torch.cat(feats, dim=0)
    probs = torch.cat(probs, dim=0)
    coords = torch.cat(coords, dim=0)
    meta = pd.concat(rows, axis=0).reset_index(drop=True)
    save_grouped_features(
        meta,
        feats,
        probs,
        coords,
        Path(args.out_dir),
        classes=classes,
        encoder_ckpt=args.ckpt,
        model_name=args.model_name,
    )


def run_scan(args, backend, classes):
    slide_dir = Path(args.slide_dir)
    out_dir = Path(args.out_dir)
    tf = T.Compose([
        T.Resize((args.tile_size, args.tile_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    for slide_path in list_wsi_paths(slide_dir):
        df = scan_slide_for_tiles(slide_path, level=args.level, tile_size=args.tile_size, step=args.step, min_tissue=args.min_tissue)
        ds = WSITileDataset(df, transform=tf, return_label=False)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        feats, probs, coords = [], [], []
        for batch in tqdm(loader, desc=f'extract-{slide_path.stem}', leave=False):
            out = backend.predict_batch(batch['image'])
            feats.append(torch.from_numpy(out.features))
            probs.append(torch.from_numpy(out.probs))
            coords.append(torch.stack([batch['x'], batch['y']], dim=1))
        blob = {
            'features': torch.cat(feats, dim=0),
            'tile_probs': torch.cat(probs, dim=0),
            'coords': torch.cat(coords, dim=0),
            'slide_id': slide_path.stem,
            'classes': list(classes),
            'encoder_ckpt': str(args.ckpt),
            'model_name': str(args.model_name),
            'levels': [int(args.level)],
            'tile_sizes': [int(args.tile_size)],
        }
        torch.save(blob, out_dir / f'{slide_path.stem}.pt')
        df.to_csv(out_dir / f'{slide_path.stem}_manifest.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['manifest', 'scan'], required=True)
    parser.add_argument('--manifest_csv', type=str, default=None)
    parser.add_argument('--slide_dir', type=str, default=None)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--backbone_pool', type=str, default='avg')
    parser.add_argument('--backbone_init_values', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--tile_size', type=int, default=224)
    parser.add_argument('--step', type=int, default=224)
    parser.add_argument('--min_tissue', type=float, default=0.4)
    parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'onnx', 'om', 'auto'])
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    backend = create_patch_backend(
        backend=args.backend,
        ckpt_path=args.ckpt,
        model_name=args.model_name,
        backbone_pool=args.backbone_pool,
        backbone_init_values=args.backbone_init_values,
        default_classes=['Normal', 'Benign', 'InSitu', 'Invasive'],
        input_size=args.tile_size,
        onnx_path=args.onnx_path,
        om_path=args.om_path,
        meta_json=args.meta_json,
        device_id=args.device_id,
    )
    try:
        if args.mode == 'manifest':
            assert args.manifest_csv is not None
            run_manifest(args, backend, backend.classes)
        else:
            assert args.slide_dir is not None
            run_scan(args, backend, backend.classes)
    finally:
        backend.close()
    print(f'features saved to {args.out_dir}')


if __name__ == '__main__':
    main()
