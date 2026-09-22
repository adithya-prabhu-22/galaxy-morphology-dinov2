import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def extract_embeddings(
    model,
    dataset,
    device,
    batch_size=64,
):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    features = []
    labels = []

    with torch.no_grad():

        for images, y in loader:

            output = model(
                images.to(device)
            )

            features.append(
                output.cpu().numpy()
            )

            labels.append(
                y.numpy()
            )

    return (
        np.concatenate(features),
        np.concatenate(labels),
    )


def train_probe(
    model,
    X_train,
    y_train,
    X_test,
    device,
    epochs=30,
    class_weights=None,
    seed=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.long,
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
        if class_weights is not None
        else None
    )

    start = time.time()

    for _ in range(epochs):

        model.train()

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            loss = loss_fn(
                model(x),
                y,
            )

            loss.backward()
            optimizer.step()

    train_time = time.time() - start

    model.eval()

    with torch.no_grad():

        predictions = (
            model(
                torch.tensor(
                    X_test,
                    dtype=torch.float32,
                ).to(device)
            )
            .argmax(1)
            .cpu()
            .numpy()
        )

    return model, predictions, train_time


def train_cnn(
    model,
    train_dataset,
    test_dataset,
    device,
    epochs=15,
    class_weights=None,
    lr=1e-4,
):
    model = model.to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
        if class_weights is not None
        else None
    )

    start = time.time()

    for _ in range(epochs):

        model.train()

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            loss = loss_fn(
                model(x),
                y,
            )

            loss.backward()
            optimizer.step()

    train_time = time.time() - start

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():

        for x, y in test_loader:

            pred = (
                model(
                    x.to(device)
                )
                .argmax(1)
                .cpu()
                .numpy()
            )

            predictions.append(pred)
            labels.append(y.numpy())

    return (
        model,
        np.concatenate(predictions),
        np.concatenate(labels),
        train_time,
    )