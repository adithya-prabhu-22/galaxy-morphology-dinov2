import os
import time
import random
import numpy as np
import pandas as pd
import torch
from galaxy_datasets import gz2

from src.dataset import (
    prepare_dataset,
    GalaxyImageDataset,
    train_transform,
    eval_transform,
)
from src.models import (
    build_dinov2,
    MLPProbe,
    LinearProbe,
    build_resnet18,
    build_dinov2_finetune,
)
from src.train import extract_embeddings, train_probe, train_cnn
from src.evaluate import evaluate


# =========================
# FINAL EXPERIMENT SETTINGS
# =========================

N_DATASET = 20000

SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

PROBE_EPOCHS = 30
CNN_EPOCHS = 15
FINETUNE_EPOCHS = 8

# Expensive models: 1%, 25%, 100%
EXPENSIVE_FRACTIONS = [0.01, 0.25, 1.00]

DATA_DIR = "data/gz2_data"
RESULTS_DIR = "results/metrics"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

for _name in [
    "dinov2_mlp",
    "dinov2_linear",
    "resnet18_scratch",
    "resnet18_imagenet",
    "dinov2_finetuned",
]:
    os.makedirs(os.path.join(MODELS_DIR, _name), exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subsample(df, fraction, seed):
    parts = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]

        n = max(1, int(len(class_df) * fraction))

        parts.append(
            class_df.sample(
                n=min(n, len(class_df)),
                random_state=seed * 1000 + int(fraction * 100),
            )
        )

    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    path = os.path.join(
        RESULTS_DIR,
        "final_results.csv"
    )

    pd.DataFrame(results).to_csv(
        path,
        index=False
    )

    return path


def run_probe_experiments(
    dino,
    train_df,
    test_dataset,
    class_weights,
    results,
):

    print("\n" + "=" * 70)
    print("DINOv2 PROBE EXPERIMENTS")
    print("=" * 70)

    train_dataset = GalaxyImageDataset(
        train_df,
        eval_transform
    )

    print("\nExtracting full DINOv2 embeddings...")

    start = time.time()

    X_train, y_train = extract_embeddings(
        dino,
        train_dataset,
        DEVICE,
        batch_size=64,
    )

    X_test, y_test = extract_embeddings(
        dino,
        test_dataset,
        DEVICE,
        batch_size=64,
    )

    print(
        "Embedding time:",
        round(time.time() - start, 2),
        "seconds"
    )

    train_labels = train_df.copy()
    train_labels["index"] = np.arange(len(train_labels))

    for fraction in FRACTIONS:

        for seed in SEEDS:

            print("\n" + "-" * 70)
            print(
                f"DINOv2 PROBES | "
                f"fraction={fraction} | seed={seed}"
            )
            print("-" * 70)

            subset_indices = []

            for label in sorted(
                train_labels["label"].unique()
            ):

                class_df = train_labels[
                    train_labels["label"] == label
                ]

                n = max(
                    1,
                    int(len(class_df) * fraction)
                )

                sampled = class_df.sample(
                    n=min(n, len(class_df)),
                    random_state=(
                        seed * 1000
                        + int(fraction * 100)
                    ),
                )

                subset_indices.extend(
                    sampled["index"].tolist()
                )

            subset_indices = np.array(
                subset_indices
            )

            X_sub = X_train[subset_indices]
            y_sub = y_train[subset_indices]

            # -------------------------
            # MLP PROBE
            # -------------------------

            set_seed(seed)

            model = MLPProbe(
                X_sub.shape[1],
                n_classes=5
            )

            model, pred, train_time = train_probe(
                model,
                X_sub,
                y_sub,
                X_test,
                DEVICE,
                epochs=PROBE_EPOCHS,
                class_weights=class_weights,
                seed=seed,
            )

            metrics = evaluate(
                y_test,
                pred
            )

            model_path = os.path.join(
                MODELS_DIR,
                "dinov2_mlp",
                f"fraction_{fraction:.2f}_seed_{seed}.pt",
            )

            torch.save(model.state_dict(), model_path)

            print(f"Saved model: {model_path}")

            results.append({
                "model": "DINOv2-MLP",
                "fraction": fraction,
                "seed": seed,
                "train_time": train_time,
                **metrics,
            })

            print(metrics)

            del model
            torch.cuda.empty_cache()

            # -------------------------
            # LINEAR PROBE
            # -------------------------

            set_seed(seed)

            model = LinearProbe(
                X_sub.shape[1],
                n_classes=5
            )

            model, pred, train_time = train_probe(
                model,
                X_sub,
                y_sub,
                X_test,
                DEVICE,
                epochs=PROBE_EPOCHS,
                class_weights=class_weights,
                seed=seed,
            )

            metrics = evaluate(
                y_test,
                pred
            )

            model_path = os.path.join(
                MODELS_DIR,
                "dinov2_linear",
                f"fraction_{fraction:.2f}_seed_{seed}.pt",
            )

            torch.save(model.state_dict(), model_path)

            print(f"Saved model: {model_path}")

            results.append({
                "model": "DINOv2-Linear",
                "fraction": fraction,
                "seed": seed,
                "train_time": train_time,
                **metrics,
            })

            print(metrics)

            del model
            torch.cuda.empty_cache()

            save_results(results)


def run_resnet_experiments(
    train_df,
    test_dataset,
    class_weights,
    results,
):

    print("\n" + "=" * 70)
    print("RESNET-18 EXPERIMENTS")
    print("=" * 70)

    for pretrained in [False, True]:

        model_name = (
            "ResNet18-ImageNet"
            if pretrained
            else "ResNet18-Scratch"
        )

        for fraction in EXPENSIVE_FRACTIONS:

            for seed in SEEDS:

                print("\n" + "-" * 70)
                print(
                    f"{model_name} | "
                    f"fraction={fraction} | seed={seed}"
                )
                print("-" * 70)

                set_seed(seed)

                subset_df = stratified_subsample(
                    train_df,
                    fraction,
                    seed
                )

                train_dataset = GalaxyImageDataset(
                    subset_df,
                    train_transform
                )

                model = build_resnet18(
                    n_classes=5,
                    pretrained=pretrained,
                ).to(DEVICE)

                model, pred, labels, train_time = train_cnn(
                    model,
                    train_dataset,
                    test_dataset,
                    DEVICE,
                    epochs=CNN_EPOCHS,
                    class_weights=class_weights,
                    lr=1e-4,
                )

                metrics = evaluate(
                    labels,
                    pred
                )

                model_dir = (
                    "resnet18_imagenet"
                    if pretrained
                    else "resnet18_scratch"
                )

                model_path = os.path.join(
                    MODELS_DIR,
                    model_dir,
                    f"fraction_{fraction:.2f}_seed_{seed}.pt",
                )

                torch.save(model.state_dict(), model_path)

                print(f"Saved model: {model_path}")

                results.append({
                    "model": model_name,
                    "fraction": fraction,
                    "seed": seed,
                    "train_time": train_time,
                    **metrics,
                })

                print(metrics)

                del model
                torch.cuda.empty_cache()

                save_results(results)


def run_dino_finetuning(
    train_df,
    test_dataset,
    class_weights,
    results,
):

    print("\n" + "=" * 70)
    print("DINOv2 FINE-TUNING")
    print("=" * 70)

    for fraction in EXPENSIVE_FRACTIONS:

        for seed in SEEDS:

            print("\n" + "-" * 70)
            print(
                f"DINOv2-Finetune | "
                f"fraction={fraction} | seed={seed}"
            )
            print("-" * 70)

            set_seed(seed)

            subset_df = stratified_subsample(
                train_df,
                fraction,
                seed
            )

            train_dataset = GalaxyImageDataset(
                subset_df,
                train_transform
            )

            model = build_dinov2_finetune(
                n_classes=5,
                unfrozen_blocks=2,
            ).to(DEVICE)

            model, pred, labels, train_time = train_cnn(
                model,
                train_dataset,
                test_dataset,
                DEVICE,
                epochs=FINETUNE_EPOCHS,
                class_weights=class_weights,
                lr=1e-5,
            )

            metrics = evaluate(
                labels,
                pred
            )

            model_path = os.path.join(
                MODELS_DIR,
                "dinov2_finetuned",
                f"fraction_{fraction:.2f}_seed_{seed}.pt",
            )

            torch.save(model.state_dict(), model_path)

            print(f"Saved model: {model_path}")

            results.append({
                "model": "DINOv2-Finetune",
                "fraction": fraction,
                "seed": seed,
                "train_time": train_time,
                **metrics,
            })

            print(metrics)

            del model
            torch.cuda.empty_cache()

            save_results(results)


def main():

    print("=" * 70)
    print("FINAL GALAXY MORPHOLOGY EXPERIMENT")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # =========================
    # DATA
    # =========================

    print("\nLoading Galaxy Zoo 2...")

    catalog, _ = gz2(
        root=DATA_DIR,
        train=True,
        download=False,
    )

    train_df, val_df, test_df, class_weights = (
        prepare_dataset(
            catalog,
            n_dataset=N_DATASET
        )
    )

    print("\nDataset ready:")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    test_dataset = GalaxyImageDataset(
        test_df,
        eval_transform
    )

    results = []

    # =========================
    # DINO BACKBONE
    # =========================

    print("\nLoading DINOv2...")

    dino = build_dinov2()
    dino = dino.to(DEVICE)
    dino.eval()

    # =========================
    # DINO PROBES
    # =========================

    run_probe_experiments(
        dino,
        train_df,
        test_dataset,
        class_weights,
        results,
    )

    del dino
    torch.cuda.empty_cache()

    # =========================
    # RESNET
    # =========================

    run_resnet_experiments(
        train_df,
        test_dataset,
        class_weights,
        results,
    )

    # =========================
    # DINO FINE-TUNING
    # =========================

    run_dino_finetuning(
        train_df,
        test_dataset,
        class_weights,
        results,
    )

    # =========================
    # FINAL OUTPUT
    # =========================

    path = save_results(results)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(path)

    df = pd.DataFrame(results)

    print("\nFinal results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

