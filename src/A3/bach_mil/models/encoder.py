from __future__ import annotations

import timm
import torch
import torch.nn as nn


class PatchClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 4,
        pretrained: bool = True,
        backbone_pool: str | None = 'avg',
        backbone_init_values: float | None = None,
    ):
        super().__init__()
        create_kwargs = {
            'pretrained': pretrained,
            'num_classes': 0,
        }
        if backbone_pool is not None:
            create_kwargs['global_pool'] = backbone_pool
        if backbone_init_values is not None:
            create_kwargs['init_values'] = backbone_init_values
        self.backbone = timm.create_model(model_name, **create_kwargs)
        feature_dim = self.backbone.num_features
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.classifier(feat)


def build_patch_model(
    model_name: str = 'vit_base_patch16_224',
    num_classes: int = 4,
    pretrained: bool = True,
    backbone_pool: str | None = 'avg',
    backbone_init_values: float | None = None,
) -> PatchClassifier:
    return PatchClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        backbone_pool=backbone_pool,
        backbone_init_values=backbone_init_values,
    )
