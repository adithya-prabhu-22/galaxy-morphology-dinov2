import torch
import torch.nn as nn
import torchvision


def build_dinov2():
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    for p in model.parameters():
        p.requires_grad = False

    return model


class MLPProbe(nn.Module):

    def __init__(
        self,
        in_dim,
        n_classes=5,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):

    def __init__(
        self,
        in_dim,
        n_classes=5,
    ):
        super().__init__()

        self.net = nn.Linear(
            in_dim,
            n_classes,
        )

    def forward(self, x):
        return self.net(x)


def build_resnet18(
    n_classes=5,
    pretrained=False,
):
    weights = (
        torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    model = torchvision.models.resnet18(
        weights=weights
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        n_classes,
    )

    return model


def build_dinov2_finetune(
    n_classes=5,
    unfrozen_blocks=2,
):
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    for p in backbone.parameters():
        p.requires_grad = False

    for block in backbone.blocks[-unfrozen_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    class DinoClassifier(nn.Module):

        def __init__(self):
            super().__init__()

            self.backbone = backbone

            self.head = nn.Linear(
                backbone.embed_dim,
                n_classes,
            )

        def forward(self, x):
            return self.head(
                self.backbone(x)
            )

    return DinoClassifier()