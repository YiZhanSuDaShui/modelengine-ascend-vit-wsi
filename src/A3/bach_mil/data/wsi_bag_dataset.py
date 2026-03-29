from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureBagDataset(Dataset):
    def __init__(
        self,
        bag_df: pd.DataFrame,
        feature_dir: str | Path,
        label_names: Sequence[str],
        max_instances: int | None = 1024,
        training: bool = True,
        instance_sampling: str = 'random',  # random | tile_prior | hybrid
    ) -> None:
        self.bag_df = bag_df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.label_names = list(label_names)
        self.max_instances = max_instances
        self.training = training
        self.instance_sampling = str(instance_sampling)

    def __len__(self) -> int:
        return len(self.bag_df)

    def __getitem__(self, index: int):
        row = self.bag_df.iloc[index]
        pt_path = self.feature_dir / f"{row.slide_id}.pt"
        blob = torch.load(pt_path, map_location='cpu')
        feats = blob['features'].float()
        coords = blob['coords'].long()
        tile_probs = blob.get('tile_probs')
        if tile_probs is not None:
            tile_probs = tile_probs.float()
        # Align tile_probs columns to current label_names when possible.
        # This lets us train 3-label MIL (e.g. drop Normal) while still using
        # a 4-class patch classifier prior saved in the feature blob.
        if tile_probs is not None and tile_probs.ndim == 2 and tile_probs.shape[0] == feats.shape[0]:
            prob_classes = blob.get('classes')
            if isinstance(prob_classes, (list, tuple)) and prob_classes and tile_probs.shape[1] != len(self.label_names):
                idx = []
                ok = True
                for name in self.label_names:
                    if name in prob_classes:
                        idx.append(int(prob_classes.index(name)))
                    else:
                        ok = False
                        break
                if ok and len(idx) == len(self.label_names):
                    tile_probs = tile_probs[:, idx]
            # Heuristic fallback: common case is 4-class probs with leading Normal.
            if tile_probs is not None and tile_probs.shape[1] == 4 and len(self.label_names) == 3 and 'Normal' not in self.label_names:
                tile_probs = tile_probs[:, 1:4]
        if self.max_instances is not None and len(feats) > self.max_instances:
            if self.instance_sampling not in {'random', 'tile_prior', 'hybrid'}:
                raise ValueError(f'Unknown instance_sampling: {self.instance_sampling}')
            if self.instance_sampling == 'random' or tile_probs is None:
                if self.training:
                    idx = torch.randperm(len(feats))[:self.max_instances]
                else:
                    idx = torch.arange(self.max_instances)
            else:
                # Use patch classifier prior to pick informative instances.
                # Exclude Normal (index 0) when present; focus on "disease-like" tiles.
                score = tile_probs[:, 1:].max(dim=1).values
                if self.instance_sampling == 'tile_prior' or not self.training:
                    idx = torch.topk(score, k=self.max_instances, largest=True).indices
                else:
                    # hybrid: half top by prior, half random from the rest
                    top_n = max(1, int(self.max_instances // 2))
                    top_idx = torch.topk(score, k=top_n, largest=True).indices
                    mask = torch.ones(len(score), dtype=torch.bool)
                    mask[top_idx] = False
                    rest = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if len(rest) > 0:
                        rand_n = self.max_instances - top_n
                        perm = torch.randperm(len(rest))[:rand_n]
                        rand_idx = rest[perm]
                        idx = torch.cat([top_idx, rand_idx], dim=0)
                    else:
                        idx = top_idx
            feats = feats[idx]
            coords = coords[idx]
            if tile_probs is not None:
                tile_probs = tile_probs[idx]
        y = torch.tensor([int(row[f'label_{k}']) for k in self.label_names], dtype=torch.float32)
        return {
            'slide_id': row.slide_id,
            'features': feats,
            'coords': coords,
            'tile_probs': tile_probs,
            'label': y,
        }


def collate_bags(batch):
    assert len(batch) == 1, 'Use batch_size=1 for bag-level MIL training.'
    return batch[0]
