from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.models.mil import ClassWiseGatedAttentionMIL
from bach_mil.utils.device import get_torch_device
from bach_mil.utils.io import ensure_dir, save_json


def _overlay_heatmap_on_rgb(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat01 = np.clip(heat01, 0.0, 1.0)
    heat_u8 = (heat01 * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=str, required=True)
    parser.add_argument('--feature_pt', type=str, required=True)
    parser.add_argument('--mil_ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--tile_level', type=int, required=True)
    parser.add_argument('--tile_size', type=int, required=True)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--thumb_max_side', type=int, default=2048)
    parser.add_argument('--alpha', type=float, default=0.45)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = get_torch_device()

    blob = torch.load(args.feature_pt, map_location='cpu')
    feats = blob['features'].float()
    coords = blob['coords'].long()
    tile_probs = blob.get('tile_probs')
    if tile_probs is not None:
        tile_probs = tile_probs.float()

    ckpt = torch.load(args.mil_ckpt, map_location='cpu')
    label_names = ckpt.get('label_names', ['Normal', 'Benign', 'InSitu', 'Invasive'])
    model = ClassWiseGatedAttentionMIL(feature_dim=args.feature_dim, num_classes=len(label_names))
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    feats_dev = feats.to(device)
    tile_probs_dev = tile_probs.to(device) if tile_probs is not None else None
    with torch.no_grad():
        out = model(feats_dev, tile_probs=tile_probs_dev)
        attn = out['attn'].detach().cpu().numpy()  # [C, N]
        prob = torch.sigmoid(out['logits']).detach().cpu().numpy()  # [C]

    # Open slide & thumbnail
    import openslide

    slide = openslide.OpenSlide(args.slide_path)
    w0, h0 = slide.dimensions
    thumb = slide.get_thumbnail((args.thumb_max_side, args.thumb_max_side)).convert('RGB')
    thumb_np = np.asarray(thumb)
    ht, wt = thumb_np.shape[:2]
    sx = float(wt) / float(w0)
    sy = float(ht) / float(h0)

    ds = float(slide.level_downsamples[int(args.tile_level)])
    tile_size0 = float(args.tile_size) * ds
    tile_w = max(1, int(round(tile_size0 * sx)))
    tile_h = max(1, int(round(tile_size0 * sy)))

    meta = {
        'slide_path': str(args.slide_path),
        'feature_pt': str(args.feature_pt),
        'mil_ckpt': str(args.mil_ckpt),
        'label_names': list(label_names),
        'prob': {k: float(v) for k, v in zip(label_names, prob.tolist())},
        'tile_level': int(args.tile_level),
        'tile_size': int(args.tile_size),
        'slide_dimensions': [int(w0), int(h0)],
        'thumb_dimensions': [int(wt), int(ht)],
        'thumb_scale': [sx, sy],
        'num_tiles': int(coords.shape[0]),
    }
    save_json(meta, out_dir / 'slide_attention_meta.json')

    # Render per-class attention overlays
    for ci, cname in enumerate(label_names):
        heat = np.zeros((ht, wt), dtype=np.float32)
        # Use max pooling so top-attended tiles remain visible.
        for i in range(coords.shape[0]):
            x0 = int(round(int(coords[i, 0]) * sx))
            y0 = int(round(int(coords[i, 1]) * sy))
            x1 = min(wt, x0 + tile_w)
            y1 = min(ht, y0 + tile_h)
            if x1 <= x0 or y1 <= y0:
                continue
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], float(attn[ci, i]))
        if float(heat.max()) > 0:
            heat01 = heat / float(heat.max())
        else:
            heat01 = heat
        overlay = _overlay_heatmap_on_rgb(thumb_np, heat01, alpha=float(args.alpha))
        Image.fromarray(overlay).save(out_dir / f'attn_{cname}.png')
        # Save heatmap itself for debugging.
        np.save(out_dir / f'attn_{cname}.npy', heat01.astype(np.float32))

    slide.close()
    print(f'saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

