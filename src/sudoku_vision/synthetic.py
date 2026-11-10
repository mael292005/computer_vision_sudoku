"""Génération de chiffres synthétiques à partir de polices de caractères.

Plutôt que d'annoter des milliers de cases à la main, on dessine nous-mêmes
des cases de Sudoku : un chiffre dans une police donnée, avec des variations
aléatoires (taille, position, rotation, épaisseur, flou, bruit). Chaque case
passe ensuite par :func:`extract_digit`, exactement comme une vraie case :
l'entraînement et l'utilisation voient des images préparées de la même façon.

Ce module nécessite Pillow (option ``ml`` ou ``dev``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from sudoku_vision.digits import extract_digit
from sudoku_vision.grid import Image

CELL_SIZE = 64
"""Côté, en pixels, des cases dessinées."""

# Polices à empattements proches de celle de websudoku.com, puis quelques
# polices différentes pour que le CNN ne dépende pas d'une seule police.
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
    "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",
    "/usr/share/fonts/truetype/baskerville/GFSBaskerville.otf",
    "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
)

TEMPLATE_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "C:/Windows/Fonts/times.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
)
"""Police des modèles du template matching : une Times (comme websudoku.com)."""


def available_fonts(candidates: Sequence[str] = FONT_CANDIDATES) -> list[Path]:
    """Garde les polices réellement présentes sur la machine."""
    return [Path(p) for p in candidates if Path(p).is_file()]


@dataclass(frozen=True)
class RenderParams:
    """Paramètres d'une case dessinée."""

    font: Path
    digit: int
    size_ratio: float = 0.62
    """Hauteur du texte par rapport à la case."""
    dx: float = 0.0
    """Décalage horizontal, en part de la case."""
    dy: float = 0.0
    """Décalage vertical, en part de la case."""
    angle: float = 0.0
    """Rotation, en degrés."""
    thickness: int = 0
    """Épaississement (> 0) ou amincissement (< 0) du trait, en pixels."""
    blur: float = 0.0
    """Écart-type du flou gaussien (0 = net)."""
    noise: float = 0.0
    """Écart-type du bruit gaussien, en niveaux de gris."""


def render_cell(params: RenderParams) -> Image:
    """Dessine une case de Sudoku contenant un chiffre (noir sur fond blanc)."""
    from PIL import Image as PilImage
    from PIL import ImageDraw, ImageFont

    canvas = PilImage.new("L", (CELL_SIZE, CELL_SIZE), 255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(params.font), round(CELL_SIZE * params.size_ratio))
    text = str(params.digit)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (CELL_SIZE - (right - left)) / 2 - left + params.dx * CELL_SIZE
    y = (CELL_SIZE - (bottom - top)) / 2 - top + params.dy * CELL_SIZE
    draw.text((x, y), text, fill=0, font=font)
    cell = np.asarray(canvas, dtype=np.uint8).copy()

    if params.angle:
        matrix = cv2.getRotationMatrix2D((CELL_SIZE / 2, CELL_SIZE / 2), params.angle, 1.0)
        cell = np.asarray(
            cv2.warpAffine(cell, matrix, (CELL_SIZE, CELL_SIZE), borderValue=255), np.uint8
        )
    if params.thickness:
        kernel = np.ones((2, 2), np.uint8)
        op = cv2.erode if params.thickness > 0 else cv2.dilate  # le chiffre est noir
        cell = np.asarray(op(cell, kernel, iterations=abs(params.thickness)), np.uint8)
    if params.blur:
        cell = np.asarray(cv2.GaussianBlur(cell, (0, 0), params.blur), np.uint8)
    if params.noise:
        rng = np.random.default_rng(params.digit * 7919 + int(params.noise * 100))
        noisy = cell.astype(np.float32) + rng.normal(0, params.noise, cell.shape)
        cell = np.clip(noisy, 0, 255).astype(np.uint8)
    return np.asarray(cell, dtype=np.uint8)


def random_params(rng: np.random.Generator, fonts: Sequence[Path], digit: int) -> RenderParams:
    """Tire des paramètres de dessin aléatoires (augmentation de données)."""
    return RenderParams(
        font=fonts[int(rng.integers(len(fonts)))],
        digit=digit,
        size_ratio=float(rng.uniform(0.45, 0.72)),
        dx=float(rng.uniform(-0.06, 0.06)),
        dy=float(rng.uniform(-0.06, 0.06)),
        angle=float(rng.uniform(-6, 6)),
        thickness=int(rng.choice([-1, 0, 0, 0, 1])),
        blur=float(rng.choice([0.0, 0.0, 0.5])),
        noise=float(rng.choice([0.0, 0.0, 5.0, 10.0])),
    )


def is_plausible_digit(normalized: Image) -> bool:
    """Filtre qualité des exemples synthétiques.

    Les chiffres 1 à 9 sont toujours plus hauts que larges : après
    normalisation, ils occupent donc toute la hauteur de la boîte de 20 px.
    Un exemple plus plat est un rendu raté (trait cassé par le flou, par
    exemple) qui apprendrait n'importe quoi au CNN : on l'écarte.
    """
    rows = np.flatnonzero(normalized.any(axis=1))
    return len(rows) > 0 and int(rows[-1] - rows[0]) + 1 >= 18


def make_dataset(
    samples_per_digit: int,
    fonts: Sequence[Path],
    seed: int = 0,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Génère un jeu de chiffres normalisés 28x28 et leurs étiquettes (0 à 8 pour 1 à 9).

    Raises:
        ValueError: si aucune police n'est fournie.
    """
    if not fonts:
        raise ValueError("aucune police disponible pour générer les chiffres")
    rng = np.random.default_rng(seed)
    images: list[Image] = []
    labels: list[int] = []
    for digit in range(1, 10):
        kept, attempts = 0, 0
        while kept < samples_per_digit and attempts < 3 * samples_per_digit:
            attempts += 1
            normalized = extract_digit(render_cell(random_params(rng, fonts, digit)))
            if normalized is not None and is_plausible_digit(normalized):
                images.append(normalized)
                labels.append(digit - 1)
                kept += 1
    return np.stack(images), np.array(labels, dtype=np.int64)


def make_templates(font: Path) -> npt.NDArray[np.float32]:
    """Construit les 9 modèles du template matching à partir d'une police.

    Chaque modèle est la moyenne de quelques rendus de taille légèrement
    différente, ce qui le rend moins sensible au redimensionnement.
    """
    templates = []
    for digit in range(1, 10):
        renders = []
        for ratio in (0.55, 0.62, 0.7):
            normalized = extract_digit(render_cell(RenderParams(font, digit, size_ratio=ratio)))
            if normalized is None:  # pragma: no cover - une police valide dessine toujours
                raise ValueError(f"impossible de dessiner le chiffre {digit} avec {font}")
            renders.append(normalized.astype(np.float32))
        templates.append(np.mean(renders, axis=0))
    return np.stack(templates).astype(np.float32)
