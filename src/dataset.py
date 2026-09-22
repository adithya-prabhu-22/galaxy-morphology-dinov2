import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = [
    "Round smooth",
    "In-between smooth",
    "Cigar-shaped smooth",
    "Edge-on disk",
    "Spiral",
]

N_CLASSES = 5
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_col(name, columns):
    """
    Find an exact GZ2 column by substring matching.
    """
    name = name.lower()

    for column in columns:
        if name in column.lower():
            return column

    return None


def prepare_dataset(catalog, n_dataset=20000):
    """
    Convert the GZ2 catalog into the five morphology classes
    used by the experiment.
    """

    columns = list(catalog.columns)

    # Actual GZ2 columns
    smooth = find_col(
        "smooth-or-featured-gz2_smooth",
        columns,
    )

    round_col = find_col(
        "how-rounded-gz2_round",
        columns,
    )

    inbetween = find_col(
        "how-rounded-gz2_in-between",
        columns,
    )

    cigar = find_col(
        "how-rounded-gz2_cigar",
        columns,
    )

    edgeon = find_col(
        "disk-edge-on-gz2_yes",
        columns,
    )

    spiral = find_col(
        "has-spiral-arms-gz2_yes",
        columns,
    )

    required = [
        smooth,
        round_col,
        inbetween,
        cigar,
        edgeon,
        spiral,
    ]

    if any(column is None for column in required):
        raise ValueError(
            "Could not resolve required GZ2 columns.\n"
            f"smooth={smooth}\n"
            f"round={round_col}\n"
            f"inbetween={inbetween}\n"
            f"cigar={cigar}\n"
            f"edgeon={edgeon}\n"
            f"spiral={spiral}"
        )

    print("\nGZ2 columns:")
    print("  Smooth:", smooth)
    print("  Round:", round_col)
    print("  In-between:", inbetween)
    print("  Cigar:", cigar)
    print("  Edge-on:", edgeon)
    print("  Spiral:", spiral)

    def get_label(row):
        # Smooth galaxy
        if row[smooth] >= 0.5:

            options = {
                0: row[round_col],
                1: row[inbetween],
                2: row[cigar],
            }

            label, value = max(
                options.items(),
                key=lambda item: item[1],
            )

            if value >= 0.5:
                return label

            return -1

        # Edge-on disk
        if row[edgeon] >= 0.715:
            return 3

        # Spiral
        if row[spiral] >= 0.619:
            return 4

        # Ambiguous galaxy
        return -1

    catalog = catalog.copy()

    catalog["label"] = catalog.apply(
        get_label,
        axis=1,
    )

    # Remove ambiguous galaxies
    clean = catalog[
        catalog["label"] >= 0
    ].copy()

    print("\nLabel distribution before balancing:")
    print(
        clean["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Balance the dataset
    per_class = max(
        1,
        n_dataset // N_CLASSES,
    )

    class_frames = []

    for label in range(N_CLASSES):

        class_data = clean[
            clean["label"] == label
        ]

        if len(class_data) == 0:
            raise ValueError(
                f"No samples found for class {label}: "
                f"{CLASS_NAMES[label]}"
            )

        sample_size = min(
            len(class_data),
            per_class,
        )

        class_frames.append(
            class_data.sample(
                n=sample_size,
                random_state=0,
            )
        )

    dataset = (
        __import__("pandas")
        .concat(class_frames)
        .sample(
            frac=1,
            random_state=0,
        )
        .reset_index(drop=True)
    )

    print("\nBalanced dataset:")
    print("Total:", len(dataset))
    print(
        dataset["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # 70 / 15 / 15 split
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.30,
        stratify=dataset["label"],
        random_state=0,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=0,
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(N_CLASSES),
        y=dataset["label"].values,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

    print("\nDataset split:")
    print("  Train:", len(train_df))
    print("  Validation:", len(val_df))
    print("  Test:", len(test_df))

    print("\nClass weights:")
    print(class_weights)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        class_weights,
    )


train_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])


class GalaxyImageDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(
            row["file_loc"]
        ).convert("RGB")

        image = self.transform(image)

        label = int(row["label"])

        return image, label
