"""Jeu de test : captures de plusieurs sites de Sudoku et leur vérité terrain."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sudoku_vision.solver import Grid, parse_grid

DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Sample:
    """Une capture d'écran et la grille attendue."""

    path: Path
    site: str
    level: str
    grid: Grid


def load_samples() -> list[Sample]:
    raw = json.loads((DATA_DIR / "grids.json").read_text(encoding="utf-8"))
    return [
        Sample(
            path=DATA_DIR / name,
            site=entry["site"],
            level=entry["level"],
            grid=parse_grid("".join(entry["grid"])),
        )
        for name, entry in raw.items()
        if not name.startswith("_")
    ]


SAMPLES = load_samples()
