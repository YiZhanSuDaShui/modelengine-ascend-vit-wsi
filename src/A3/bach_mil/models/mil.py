from __future__ import annotations

import torch
import torch.nn as nn


class ClassWiseGatedAttentionMIL(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.attn_v = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_u = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_w = nn.Linear(hidden_dim, num_classes)
        self.cls = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor, tile_probs: torch.Tensor | None = None):
        # features: [N, D]
        v = self.attn_v(features)
        u = self.attn_u(features)
        a = self.attn_w(v * u)  # [N, C]
        a = torch.softmax(a.transpose(0, 1), dim=1)  # [C, N]
        bag_repr = a @ features  # [C, D]
        logits = self.cls(bag_repr)  # [C, C]
        bag_logits = logits.diag()  # class-wise independent logit
        if tile_probs is not None and tile_probs.ndim == 2 and tile_probs.shape[1] == self.num_classes:
            topk = min(8, tile_probs.shape[0])
            tile_prior = torch.topk(tile_probs, k=topk, dim=0).values.mean(dim=0)
            # Avoid torch.logit() because it may fall back to CPU on some backends (e.g. NPU).
            eps = 1e-4
            prior = tile_prior.clamp(eps, 1 - eps)
            prior_logit = torch.log(prior) - torch.log1p(-prior)
            bag_logits = 0.7 * bag_logits + 0.3 * prior_logit
        return {'logits': bag_logits, 'attn': a, 'bag_repr': bag_repr}
