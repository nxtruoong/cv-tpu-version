"""ResNet-18 backbone + SimCLR projection head + supervised classifier."""
from typing import Optional, Dict
import torch
import torch.nn as nn
from torchvision import models

from .config import NUM_CLASSES, PROJECTION_DIM, PROJECTION_HIDDEN


def _build_backbone(pretrained: bool = False) -> nn.Module:
    """ResNet-18 with fc replaced by Identity. Output dim = 512."""
    if pretrained:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        net = models.resnet18(weights=None)
    feature_dim = net.fc.in_features  # 512
    net.fc = nn.Identity()
    net.feature_dim = feature_dim
    return net


class ProjectionHead(nn.Module):
    """MLP head for SimCLR: Linear -> ReLU -> Linear."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = PROJECTION_HIDDEN,
                 out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # inplace=False: safer with XLA + bf16 autograd than inplace ReLU.
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """Backbone + projection head. Returns L2-normalized projection."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = _build_backbone(pretrained=pretrained_backbone)
        self.projection = ProjectionHead(in_dim=self.backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projection(h)
        return nn.functional.normalize(z, dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features only (no projection). For linear probe / fine-tune."""
        return self.backbone(x)


class ClassifierModel(nn.Module):
    """Backbone + Linear classifier head. Used for all 3 fine-tune conditions."""

    def __init__(self, pretrained_backbone: bool = False,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = _build_backbone(pretrained=pretrained_backbone)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)


def load_simclr_backbone_into_classifier(
    classifier: ClassifierModel, simclr_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Copy SimCLR backbone weights into ClassifierModel.backbone. Discards projection."""
    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in simclr_state_dict.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = classifier.backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading SimCLR backbone: {unexpected}")
    return missing, unexpected


def freeze_backbone(model: ClassifierModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: ClassifierModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True


def discriminative_param_groups(model: ClassifierModel) -> list:
    """Per-layer LR for stage-2 fine-tune."""
    return [
        {"params": model.backbone.conv1.parameters(), "lr": 1e-5},
        {"params": model.backbone.bn1.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer1.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer2.parameters(), "lr": 5e-5},
        {"params": model.backbone.layer3.parameters(), "lr": 5e-5},
        {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]


def build_classifier_for_condition(
    condition: str, simclr_ckpt_path: Optional[str] = None
) -> ClassifierModel:
    """Build classifier for one of {A_scratch, B_simclr, C_imagenet}."""
    if condition == "A_scratch":
        return ClassifierModel(pretrained_backbone=False)
    if condition == "C_imagenet":
        return ClassifierModel(pretrained_backbone=True)
    if condition == "B_simclr":
        if simclr_ckpt_path is None:
            raise ValueError("B_simclr requires simclr_ckpt_path")
        model = ClassifierModel(pretrained_backbone=False)
        state = torch.load(simclr_ckpt_path, map_location="cpu")
        sd = state.get("model_state_dict", state)
        load_simclr_backbone_into_classifier(model, sd)
        return model
    raise ValueError(f"Unknown condition: {condition}")
