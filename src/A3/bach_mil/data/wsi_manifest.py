from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .label_map import WSI_LABELS_DEFAULT
from .xml_utils import parse_xml_polygons, rasterize_polygons
from ..utils.tissue import tissue_mask_from_rgb

WSI_SUFFIXES = ('.svs', '.tif', '.tiff')


def list_wsi_paths(wsi_dir: str | Path) -> list[Path]:
    wsi_dir = Path(wsi_dir)
    if not wsi_dir.exists():
        return []
    return sorted(
        [p for p in wsi_dir.iterdir() if p.is_file() and p.suffix.lower() in WSI_SUFFIXES],
        key=lambda p: p.name.lower(),
    )


def _resolve_tiffslide_cls():
    vendor_dir = Path(__file__).resolve().parents[4] / '_vendor'
    if vendor_dir.exists() and str(vendor_dir) not in sys.path:
        sys.path.append(str(vendor_dir))
    from tiffslide import TiffSlide
    return TiffSlide


def _format_backend_error(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        text = repr(exc)
    return text


def _resolve_slide_cls():
    try:
        import openslide
        return openslide.OpenSlide
    except Exception:
        try:
            return _resolve_tiffslide_cls()
        except Exception as tiffslide_error:
            raise ModuleNotFoundError('未找到 openslide，且 tiffslide 回退也不可用。') from tiffslide_error


def _get_slide(slide_path: str):
    slide_path = Path(slide_path)
    openslide_error = None
    try:
        slide_cls = _resolve_slide_cls()
        return slide_cls(str(slide_path))
    except Exception as exc:
        openslide_error = exc

    try:
        slide_cls = _resolve_tiffslide_cls()
        return slide_cls(str(slide_path))
    except Exception as tiffslide_error:
        raise RuntimeError(
            f'无法打开 WSI 文件：{slide_path}。'
            f'openslide 错误：{_format_backend_error(openslide_error)}；'
            f'tiffslide 错误：{_format_backend_error(tiffslide_error)}。'
        ) from tiffslide_error


def level_downsample(slide, level: int) -> float:
    return float(slide.level_downsamples[level])


def build_annotated_tile_manifest(
    wsi_dir: str | Path,
    out_csv: str | Path,
    out_bag_csv: str | Path,
    level: int = 1,
    tile_size: int = 224,
    step: int = 224,
    min_tissue: float = 0.4,
    label_names=None,
    include_normal_from_unannotated: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_names is None:
        label_names = WSI_LABELS_DEFAULT
    wsi_dir = Path(wsi_dir)
    rows = []
    bag_rows = []
    for svs_path in sorted(wsi_dir.glob('A*.svs')):
        xml_path = svs_path.with_suffix('.xml')
        if not xml_path.exists():
            continue
        slide_id = svs_path.stem
        slide = _get_slide(svs_path)
        dims = slide.level_dimensions[level]
        ds = level_downsample(slide, level)
        thumb = slide.read_region((0, 0), level, dims).convert('RGB')
        thumb_np = np.asarray(thumb)
        tissue = tissue_mask_from_rgb(thumb_np)
        polygons = parse_xml_polygons(xml_path)
        anno_mask = rasterize_polygons(polygons, out_hw=(dims[1], dims[0]), downsample=ds, label_names=label_names)

        slide_labels = {name: 0 for name in label_names}
        for poly in polygons:
            if poly['label_name'] in slide_labels:
                slide_labels[poly['label_name']] = 1

        h, w = tissue.shape
        for y in range(0, max(1, h - tile_size + 1), step):
            for x in range(0, max(1, w - tile_size + 1), step):
                tissue_patch = tissue[y:y + tile_size, x:x + tile_size]
                if tissue_patch.size == 0:
                    continue
                tissue_ratio = float(tissue_patch.mean())
                if tissue_ratio < min_tissue:
                    continue
                anno_patch = anno_mask[y:y + tile_size, x:x + tile_size]
                label_name = None
                if anno_patch.size > 0 and anno_patch.max() > 0:
                    vals, counts = np.unique(anno_patch[anno_patch > 0], return_counts=True)
                    label_idx = int(vals[np.argmax(counts)]) - 1
                    label_name = label_names[label_idx]
                elif include_normal_from_unannotated and 'Normal' in label_names:
                    label_name = 'Normal'
                if label_name is None:
                    continue
                x0 = int(round(x * ds))
                y0 = int(round(y * ds))
                rows.append({
                    'slide_id': slide_id,
                    'slide_path': str(svs_path),
                    'xml_path': str(xml_path),
                    'x': x0,
                    'y': y0,
                    'level': level,
                    'tile_size': tile_size,
                    'label_name': label_name,
                    'source': 'wsi_xml',
                })
        bag_row = {'slide_id': slide_id, 'slide_path': str(svs_path)}
        bag_row.update({f'label_{k}': int(v) for k, v in slide_labels.items()})
        bag_rows.append(bag_row)
        slide.close()

    tiles_df = pd.DataFrame(rows)
    bags_df = pd.DataFrame(bag_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tiles_df.to_csv(out_csv, index=False)
    bags_df.to_csv(out_bag_csv, index=False)
    return tiles_df, bags_df


def scan_slide_for_tiles(
    slide_path: str | Path,
    level: int = 1,
    tile_size: int = 224,
    step: int = 224,
    min_tissue: float = 0.4,
) -> pd.DataFrame:
    slide_path = Path(slide_path)
    slide = _get_slide(slide_path)
    dims = slide.level_dimensions[level]
    ds = float(slide.level_downsamples[level])
    thumb = slide.read_region((0, 0), level, dims).convert('RGB')
    tissue = tissue_mask_from_rgb(np.asarray(thumb))
    h, w = tissue.shape
    rows = []
    for y in range(0, max(1, h - tile_size + 1), step):
        for x in range(0, max(1, w - tile_size + 1), step):
            tissue_patch = tissue[y:y + tile_size, x:x + tile_size]
            if tissue_patch.size == 0:
                continue
            tissue_ratio = float(tissue_patch.mean())
            if tissue_ratio < min_tissue:
                continue
            rows.append({
                'slide_id': slide_path.stem,
                'slide_path': str(slide_path),
                'x': int(round(x * ds)),
                'y': int(round(y * ds)),
                'level': level,
                'tile_size': tile_size,
                'label_name': 'Unknown',
                'source': 'scan',
            })
    slide.close()
    return pd.DataFrame(rows)


class WSITileDataset:
    def __init__(self, df: pd.DataFrame, transform=None, return_label: bool = True, label_names=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_label = return_label
        self.label_names = label_names or WSI_LABELS_DEFAULT
        self.label_to_idx = {k: i for i, k in enumerate(self.label_names)}
        self._slide_cache = {}

    def __len__(self):
        return len(self.df)

    def _open(self, slide_path: str):
        if slide_path not in self._slide_cache:
            self._slide_cache[slide_path] = _get_slide(slide_path)
        return self._slide_cache[slide_path]

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        slide = self._open(row.slide_path)
        img = slide.read_region((int(row.x), int(row.y)), int(row.level), (int(row.tile_size), int(row.tile_size))).convert('RGB')
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = img
        out = {
            'image': x,
            'slide_id': row.slide_id,
            'x': int(row.x),
            'y': int(row.y),
            'level': int(row.level),
            'tile_size': int(row.tile_size),
            'label_name': row.label_name,
        }
        if self.return_label and row.label_name in self.label_to_idx:
            out['label'] = self.label_to_idx[row.label_name]
        return out
