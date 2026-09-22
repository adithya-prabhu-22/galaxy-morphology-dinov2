import torch

from src.models import (
    build_dinov2,
    MLPProbe,
    LinearProbe,
    build_resnet18,
    build_dinov2_finetune,
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 50)
    print("Galaxy Morphology - GPU Sanity Test")
    print("=" * 50)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "CUDA:",
            torch.version.cuda,
        )

    # ----------------------------------------
    # Test DINOv2
    # ----------------------------------------

    print("\nLoading DINOv2...")

    dino = build_dinov2()

    dino = dino.to(device)

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        features = dino(x)

    print(
        "DINOv2 output:",
        features.shape,
    )

    # ----------------------------------------
    # Test probes
    # ----------------------------------------

    mlp = MLPProbe(
        features.shape[1]
    ).to(device)

    linear = LinearProbe(
        features.shape[1]
    ).to(device)

    print(
        "MLP output:",
        mlp(features).shape,
    )

    print(
        "Linear output:",
        linear(features).shape,
    )

    # ----------------------------------------
    # Test ResNet
    # ----------------------------------------

    resnet = build_resnet18(
        n_classes=5,
        pretrained=False,
    )

    print(
        "ResNet output:",
        resnet(x).shape,
    )

    # ----------------------------------------
    # Test fine-tuned DINO
    # ----------------------------------------

    print("\nLoading fine-tuning model...")

    dino_ft = build_dinov2_finetune(
        n_classes=5
    ).to(device)

    print(
        "DINOv2 fine-tuning output:",
        dino_ft(x).shape,
    )

    print("\nAll model tests passed.")
    print("=" * 50)


if __name__ == "__main__":
    main()