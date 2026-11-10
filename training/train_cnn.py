"""Entraîne le CNN de reconnaissance des chiffres et l'exporte en ONNX.

Le modèle n'est entraîné que sur des chiffres synthétiques : une cinquantaine
de polices (celles du système et les polices web de training/fonts.py), en
graisses variées, avec des déformations aléatoires. Les captures de tests/data
ne servent qu'à l'évaluation finale.

Usage :
    pip install -e ".[ml]"
    python training/train_cnn.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from fonts import download_fonts
from torch import nn

from sudoku_vision.digits import CNN_PATH
from sudoku_vision.synthetic import available_fonts, make_dataset


class DigitCnn(nn.Module):
    """Deux blocs convolution + pooling, puis deux couches denses. ~60 000 paramètres."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-digit", type=int, default=3000, help="exemples par chiffre")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    fonts = available_fonts() + download_fonts()
    print(f"{len(fonts)} polices : {', '.join(f.stem for f in fonts)}")

    start = time.perf_counter()
    x_train, y_train = make_dataset(args.per_digit, fonts, seed=args.seed)
    x_val, y_val = make_dataset(args.per_digit // 5, fonts, seed=args.seed + 1)
    print(
        f"Données : {len(x_train)} entraînement, {len(x_val)} validation "
        f"({time.perf_counter() - start:.1f} s)"
    )

    xt, yt = to_tensor(x_train), torch.from_numpy(y_train)
    xv, yv = to_tensor(x_val), torch.from_numpy(y_val)

    model = DigitCnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 128

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        print(
            f"Époque {epoch:2d}  perte {total / len(xt):.4f}  "
            f"validation {accuracy(model, xv, yv):.2%}"
        )

    model.eval()
    CNN_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 28, 28),),
        str(CNN_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Modèle exporté ({params} paramètres) -> {CNN_PATH}")


if __name__ == "__main__":
    main()
