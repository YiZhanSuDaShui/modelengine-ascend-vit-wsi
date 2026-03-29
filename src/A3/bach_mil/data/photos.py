from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .label_map import PHOTO_TO_INDEX, PHOTO_LABELS
from ..utils.tissue import pil_to_np, choose_tissue_crop


@dataclass
class PhotoRecord:
    image_path: str
    label_name: str
    fold: int = -1
    split: str = 'train'


def scan_photos(photos_root: str | Path) -> pd.DataFrame:
    photos_root = Path(photos_root)
    rows = []
    for label_name in PHOTO_LABELS:
        sub = photos_root / label_name
        if not sub.exists():
            continue
        for path in sorted(sub.glob('*.tif')):
            rows.append({
                'image_path': str(path),
                'image_id': path.stem,
                'label_name': label_name,
                'label': PHOTO_TO_INDEX[label_name],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f'No tif files found under {photos_root}')
    return df


class PhotoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int = 224,
        training: bool = True,
        tissue_crop: bool = False,
        min_tissue: float = 0.3,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.input_size = input_size
        self.training = training
        self.tissue_crop = tissue_crop
        self.min_tissue = min_tissue
        if training:
            self.aug = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)], p=0.5),
                T.RandomErasing(p=0.1),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.aug = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = self._load(row.image_path)
        if self.training:
            if self.tissue_crop:
                arr = pil_to_np(img)
                crop = choose_tissue_crop(arr, crop_size=min(arr.shape[0], arr.shape[1], 512), min_tissue=self.min_tissue)
                img = Image.fromarray(crop)
            trans = T.RandomResizedCrop(size=self.input_size, scale=(0.5, 1.0), ratio=(0.75, 1.33))
            img = trans(img)
        else:
            img = T.Resize((self.input_size, self.input_size))(img)
        x = self.aug(img)
        y = int(row.label)
        return {'image': x, 'label': y, 'image_id': row.image_id, 'image_path': row.image_path}
