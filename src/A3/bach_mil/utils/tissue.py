from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert('RGB'))


def tissue_mask_from_rgb(rgb: np.ndarray, sat_thresh: int = 15, gray_thresh: int = 235) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (sat > sat_thresh) & (gray < gray_thresh)
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def tissue_ratio(rgb: np.ndarray) -> float:
    m = tissue_mask_from_rgb(rgb)
    return float(m.mean())


def choose_tissue_crop(rgb: np.ndarray, crop_size: int, min_tissue: float = 0.3, tries: int = 20) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        rgb = cv2.resize(rgb, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
    for _ in range(tries):
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        crop = rgb[y:y + crop_size, x:x + crop_size]
        if tissue_ratio(crop) >= min_tissue:
            return crop
    y = max(0, (h - crop_size) // 2)
    x = max(0, (w - crop_size) // 2)
    return rgb[y:y + crop_size, x:x + crop_size]
