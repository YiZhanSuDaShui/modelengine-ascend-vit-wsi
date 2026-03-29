from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

from .tissue import tissue_mask_from_rgb


@dataclass(frozen=True)
class PatchAggConfig:
    input_size: int = 224
    crop_sizes: tuple[int, ...] = (512,)
    stride: int = 256
    topk_per_size: int = 8
    min_tissue: float = 0.4
    working_max_side: int = 1024
    agg: str = 'topk_mean_logit'  # mean_prob | max_prob | topk_mean_prob | topk_mean_logit
    logit_topk: int = 8


def _to_working_rgb(rgb: np.ndarray, working_max_side: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    max_side = max(h, w)
    if max_side <= working_max_side:
        return rgb, 1.0
    scale = float(working_max_side) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _integral(mask01: np.ndarray) -> np.ndarray:
    # cv2.integral returns shape [H+1, W+1]
    return cv2.integral(mask01, sdepth=cv2.CV_64F)


def _window_sum(ii: np.ndarray, y: int, x: int, h: int, w: int) -> float:
    y1 = y + h
    x1 = x + w
    return float(ii[y1, x1] - ii[y, x1] - ii[y1, x] + ii[y, x])


def _pick_topk_windows(
    tissue_mask01: np.ndarray,
    crop_size: int,
    stride: int,
    topk: int,
    min_tissue: float,
) -> list[tuple[float, int, int]]:
    h, w = tissue_mask01.shape[:2]
    crop_size = int(min(crop_size, h, w))
    stride = int(max(1, min(stride, crop_size)))
    ii = _integral(tissue_mask01)
    cand: list[tuple[float, int, int]] = []
    area = float(crop_size * crop_size)
    for y in range(0, max(1, h - crop_size + 1), stride):
        for x in range(0, max(1, w - crop_size + 1), stride):
            s = _window_sum(ii, y, x, crop_size, crop_size)
            ratio = s / area
            if ratio >= min_tissue:
                cand.append((ratio, x, y))
    if not cand:
        # fallback to center window even if tissue is low
        y = max(0, (h - crop_size) // 2)
        x = max(0, (w - crop_size) // 2)
        ratio = _window_sum(ii, y, x, crop_size, crop_size) / area
        return [(ratio, x, y)]
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[: max(1, int(topk))]


def sample_tissue_crops(
    rgb: np.ndarray,
    crop_sizes: Sequence[int],
    stride: int,
    topk_per_size: int,
    min_tissue: float,
    working_max_side: int = 1024,
) -> tuple[list[np.ndarray], dict]:
    """Sample tissue-rich crops from an RGB image at multiple crop sizes.

    Returns a list of crops (np.uint8 RGB) and a meta dict for debugging.
    """
    work_rgb, scale = _to_working_rgb(rgb, working_max_side=working_max_side)
    work_mask = tissue_mask_from_rgb(work_rgb).astype(np.uint8)

    h0, w0 = rgb.shape[:2]
    hs, ws = work_rgb.shape[:2]
    crops: list[np.ndarray] = []
    meta_windows = []

    for crop_sz0 in crop_sizes:
        crop_sz0 = int(min(crop_sz0, h0, w0))
        if crop_sz0 <= 0:
            continue
        crop_sz = int(max(1, round(crop_sz0 * scale)))
        stride_s = int(max(1, round(stride * scale)))
        wins = _pick_topk_windows(
            tissue_mask01=work_mask,
            crop_size=crop_sz,
            stride=stride_s,
            topk=topk_per_size,
            min_tissue=min_tissue,
        )
        for ratio, xs, ys in wins:
            # Map working coords back to original coords.
            xo = int(round(xs / scale)) if scale > 0 else int(xs)
            yo = int(round(ys / scale)) if scale > 0 else int(ys)
            xo = int(np.clip(xo, 0, max(0, w0 - crop_sz0)))
            yo = int(np.clip(yo, 0, max(0, h0 - crop_sz0)))
            crop = rgb[yo:yo + crop_sz0, xo:xo + crop_sz0]
            if crop.size == 0:
                continue
            crops.append(crop)
            meta_windows.append({
                'crop_size': int(crop_sz0),
                'x': int(xo),
                'y': int(yo),
                'tissue_ratio': float(ratio),
                'work_x': int(xs),
                'work_y': int(ys),
                'work_crop': int(crop_sz),
            })

    if not crops:
        # Absolute fallback: use whole image resized.
        crops = [rgb]
        meta_windows = [{
            'crop_size': int(min(h0, w0)),
            'x': 0,
            'y': 0,
            'tissue_ratio': float(work_mask.mean()),
            'work_x': 0,
            'work_y': 0,
            'work_crop': int(min(hs, ws)),
        }]

    meta = {
        'num_crops': int(len(crops)),
        'working_scale': float(scale),
        'working_hw': [int(hs), int(ws)],
        'orig_hw': [int(h0), int(w0)],
        'windows': meta_windows,
    }
    return crops, meta


def infer_image_patch_agg(
    model,
    img: Image.Image,
    device: str,
    num_classes: int,
    cfg: PatchAggConfig,
    batch_size: int = 64,
) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(img.convert('RGB'))
    crops, meta = sample_tissue_crops(
        rgb,
        crop_sizes=list(cfg.crop_sizes),
        stride=cfg.stride,
        topk_per_size=cfg.topk_per_size,
        min_tissue=cfg.min_tissue,
        working_max_side=cfg.working_max_side,
    )

    tf = T.Compose([
        T.Resize((cfg.input_size, cfg.input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    probs = []
    logits_all = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = [tf(Image.fromarray(c)) for c in crops[i:i + batch_size]]
            x = torch.stack(batch, dim=0).to(device)
            logit = model(x).detach().cpu().numpy()
            if cfg.agg != 'topk_mean_logit':
                # Only compute per-crop softmax probabilities when needed.
                p = torch.softmax(torch.from_numpy(logit), dim=1).numpy()
                probs.append(p)
            logits_all.append(logit)
    logits_np = np.concatenate(logits_all, axis=0)

    if cfg.agg == 'mean_prob':
        probs_np = np.concatenate(probs, axis=0)
        prob = probs_np.mean(axis=0)
    elif cfg.agg == 'max_prob':
        probs_np = np.concatenate(probs, axis=0)
        prob = probs_np.max(axis=0)
    elif cfg.agg == 'topk_mean_prob':
        probs_np = np.concatenate(probs, axis=0)
        k = int(max(1, min(cfg.logit_topk, probs_np.shape[0])))
        # top-k per class on probabilities, then mean
        topk = np.partition(probs_np, kth=probs_np.shape[0] - k, axis=0)[-k:, :]
        prob = topk.mean(axis=0)
    elif cfg.agg == 'topk_mean_logit':
        k = int(max(1, min(cfg.logit_topk, logits_np.shape[0])))
        topk = np.partition(logits_np, kth=logits_np.shape[0] - k, axis=0)[-k:, :]
        bag_logit = topk.mean(axis=0)
        # stable softmax
        bag_logit = bag_logit - bag_logit.max()
        exp = np.exp(bag_logit)
        prob = exp / exp.sum()
    else:
        raise ValueError(f'Unknown cfg.agg: {cfg.agg}')
    assert prob.shape == (num_classes,)
    return prob, meta


def infer_image_patch_agg_backend(
    backend: Any,
    img: Image.Image,
    num_classes: int,
    cfg: PatchAggConfig,
    batch_size: int = 64,
    return_crop_outputs: bool = False,
) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(img.convert('RGB'))
    crops, meta = sample_tissue_crops(
        rgb,
        crop_sizes=list(cfg.crop_sizes),
        stride=cfg.stride,
        topk_per_size=cfg.topk_per_size,
        min_tissue=cfg.min_tissue,
        working_max_side=cfg.working_max_side,
    )

    tf = T.Compose([
        T.Resize((cfg.input_size, cfg.input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    probs = []
    logits_all = []
    features_all = []
    for i in range(0, len(crops), batch_size):
        batch = [tf(Image.fromarray(c)) for c in crops[i:i + batch_size]]
        x = torch.stack(batch, dim=0)
        out = backend.predict_batch(x)
        logit = out.logits
        if cfg.agg != 'topk_mean_logit':
            p = np.exp(logit - logit.max(axis=1, keepdims=True))
            p = p / np.clip(p.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
            probs.append(p)
        logits_all.append(logit)
        if return_crop_outputs:
            features_all.append(out.features)
    logits_np = np.concatenate(logits_all, axis=0)

    if cfg.agg == 'mean_prob':
        probs_np = np.concatenate(probs, axis=0)
        prob = probs_np.mean(axis=0)
    elif cfg.agg == 'max_prob':
        probs_np = np.concatenate(probs, axis=0)
        prob = probs_np.max(axis=0)
    elif cfg.agg == 'topk_mean_prob':
        probs_np = np.concatenate(probs, axis=0)
        k = int(max(1, min(cfg.logit_topk, probs_np.shape[0])))
        topk = np.partition(probs_np, kth=probs_np.shape[0] - k, axis=0)[-k:, :]
        prob = topk.mean(axis=0)
    elif cfg.agg == 'topk_mean_logit':
        k = int(max(1, min(cfg.logit_topk, logits_np.shape[0])))
        topk = np.partition(logits_np, kth=logits_np.shape[0] - k, axis=0)[-k:, :]
        bag_logit = topk.mean(axis=0)
        bag_logit = bag_logit - bag_logit.max()
        exp = np.exp(bag_logit)
        prob = exp / exp.sum()
    else:
        raise ValueError(f'Unknown cfg.agg: {cfg.agg}')

    if return_crop_outputs:
        meta['crop_logits'] = logits_np.astype(np.float32, copy=False)
        meta['crop_features'] = np.concatenate(features_all, axis=0).astype(np.float32, copy=False)

    assert prob.shape == (num_classes,)
    return prob.astype(np.float32, copy=False), meta
