from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .label_map import normalize_xml_label, WSI_LABELS_DEFAULT


def _tag_name(tag: str) -> str:
    # XML files may contain namespace prefixes; normalize to local lowercase tag.
    return tag.split('}', 1)[-1].lower()


def _parse_annotation_coords(node) -> list[tuple[float, float]]:
    coords = []
    for child in node.iter():
        if _tag_name(child.tag) == 'coordinate':
            x = child.attrib.get('X') or child.attrib.get('x')
            y = child.attrib.get('Y') or child.attrib.get('y')
            if x is not None and y is not None:
                coords.append((float(x), float(y)))
    return coords


def _parse_region_vertices(node) -> list[tuple[float, float]]:
    coords = []
    for child in node.iter():
        if _tag_name(child.tag) == 'vertex':
            x = child.attrib.get('X') or child.attrib.get('x')
            y = child.attrib.get('Y') or child.attrib.get('y')
            if x is not None and y is not None:
                coords.append((float(x), float(y)))
    return coords


def _region_label(region) -> str | None:
    # Aperio-style XML usually stores class string in Region/Attributes/Attribute@Value.
    attrs = region.find('./Attributes')
    if attrs is not None:
        for attr in attrs:
            if _tag_name(attr.tag) != 'attribute':
                continue
            raw = attr.attrib.get('Value') or attr.attrib.get('Name')
            label = normalize_xml_label(raw)
            if label is not None:
                return label
    # Fallback: try region-level text fields.
    for key in ('Text', 'Name', 'Type'):
        raw = region.attrib.get(key)
        label = normalize_xml_label(raw)
        if label is not None:
            return label
    return None


def parse_xml_polygons(xml_path: str | Path) -> List[Dict[str, Any]]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    polygons = []

    # 1) QuPath-like format: Annotation + Coordinate points.
    for ann in root.iter():
        if _tag_name(ann.tag) != 'annotation':
            continue
        raw_label = ann.attrib.get('PartOfGroup') or ann.attrib.get('Name') or ann.attrib.get('Type')
        label = normalize_xml_label(raw_label)
        coords = _parse_annotation_coords(ann)
        if label is not None and len(coords) >= 3:
            polygons.append({'label_name': label, 'points_xy': coords})

    # 2) Aperio-like format: Region + Vertex points + Attribute Value labels.
    for region in root.iter():
        if _tag_name(region.tag) != 'region':
            continue
        label = _region_label(region)
        coords = _parse_region_vertices(region)
        if label is not None and len(coords) >= 3:
            polygons.append({'label_name': label, 'points_xy': coords})
    return polygons


def rasterize_polygons(polygons, out_hw, downsample: float, label_names=None) -> np.ndarray:
    if label_names is None:
        label_names = WSI_LABELS_DEFAULT
    label_to_idx = {k: i + 1 for i, k in enumerate(label_names)}
    h, w = out_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        label_name = poly['label_name']
        if label_name not in label_to_idx:
            continue
        pts = np.array(poly['points_xy'], dtype=np.float32) / float(downsample)
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], color=int(label_to_idx[label_name]))
    return mask
