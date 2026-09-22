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


def find_col(candidates, cols):
    for c in cols:
        name = c.lower()
        if all(x in name for x in candidates):
            return c
    return None


def prepare_dataset(catalog, n_dataset=20000):
    cols = list(catalog.columns)

    smooth = find_col(["smooth-or-featured", "smooth", "fraction"], cols)
    round_col = find_col(["round", "fraction"], cols)
    inbetween = (
        find_col(["in-between", "fraction"], cols)
        or find_col(["inbetween", "fraction"], cols)
    )
    cigar = find_col(["cigar", "fraction"], cols)
    edgeon = (
        find_col(["edge-on", "yes", "fraction"], cols)
        or find_col(["edgeon", "yes", "fraction"], cols)
    )
    spiral = (
        find_col(["spiral-arms", "yes", "fraction"], cols)
        or find_col(["has-spiral", "yes", "fraction"], cols)
    )

    vote_count = (
        find_col(["smooth-or-featured", "count"], cols)
        or find_col(["smooth-or-featured", "total-votes"], cols)
        or find_col(["smooth-or-featured", "total_votes"], cols)
    )

    required = [smooth, round_col, inbetween, cigar, edgeon, spiral]

    if any(x is None for x in required):
        raise ValueError("Could not resolve required GZ2 columns.")

    if vote_count:
        catalog = catalog[catalog[vote_count] >= 10].copy()

    def get_label(row):
        if row[smooth] >= 0.5:
            options = {
                0: row[round_col],
                1: row[inbetween],
                2: row[cigar],
            }

            label, value = max(
                options.items(),
                key=lambda x: x[1],
            )

            return label if value >= 0.5 else -1

        if row[edgeon] >= 0.715:
            return 3

        if row[spiral] >= 0.619:
            return 4

        return -1

    catalog["label"] = catalog.apply(
        get_label,
        axis=1,
    )

    clean = catalog[catalog["label"] >= 0].copy()

    per_class = max(
        1,
        n_dataset // N_CLASSES,
    )

    dataset = (
        clean.groupby("label", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), per_class),
                random_state=0,
            )
        )
        .reset_index(drop=True)
    )

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

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(N_CLASSES),
        y=dataset["label"].values,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        torch.tensor(
            class_weights,
            dtype=torch.float32,
        ),
    )


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

        return image, int(row["label"])