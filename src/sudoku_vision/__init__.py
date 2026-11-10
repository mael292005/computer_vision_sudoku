"""Sudoku Vision : lire une grille de Sudoku à l'écran, la résoudre et la remplir."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sudoku-vision")
except PackageNotFoundError:  # pragma: no cover - uniquement depuis un dépôt non installé
    __version__ = "0.0.0"

__all__ = ["__version__"]
