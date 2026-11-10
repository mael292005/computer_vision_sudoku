"""Point central de tous les paramètres réglables du pipeline.

Regrouper les seuils et les tailles ici (plutôt que des « valeurs magiques »
éparpillées dans le code) rend les étapes de vision plus faciles à lire,
à tester et à ajuster.
"""

from __future__ import annotations

from dataclasses import dataclass

GRID_SIZE = 9
"""Nombre de lignes et de colonnes d'une grille de Sudoku standard."""

BOX_SIZE = 3
"""Nombre de lignes et de colonnes d'un bloc 3x3."""


@dataclass(frozen=True)
class VisionConfig:
    """Paramètres des étapes de traitement d'image."""

    warped_size: int = 450
    """Côté, en pixels, de l'image redressée (vue de face) de la grille."""

    cell_margin_ratio: float = 0.12
    """Part de chaque case rognée sur chaque bord pour retirer les lignes de la grille."""

    empty_cell_ink_ratio: float = 0.01
    """Sous cette proportion de pixels « encre », la case est considérée vide.

    Mesuré sur websudoku.com : 0 % pour une case vide, au moins 3,7 % pour un
    chiffre (le « 1 », le plus fin). Le seuil est placé loin des deux.
    """

    digit_size: int = 28
    """Côté, en pixels, de l'image de chiffre donnée au classifieur (format MNIST)."""

    min_digit_height_ratio: float = 0.2
    """Hauteur minimale de l'encre, en part de la case, pour qu'il s'agisse d'un chiffre.

    Mesuré : au moins 39 % pour un vrai chiffre, au plus 8 % pour une poussière ou
    un reflet sur une photo d'écran. Sans ce seuil, un simple point était agrandi
    à la taille d'un chiffre et le CNN inventait un « 4 ».
    """


@dataclass(frozen=True)
class ActuatorConfig:
    """Paramètres de l'automatisation souris et clavier."""

    pause_seconds: float = 0.05
    """Délai entre deux actions, pour laisser le site suivre."""

    failsafe: bool = True
    """Amener la souris dans un coin de l'écran arrête immédiatement le bot."""


DEFAULT_VISION = VisionConfig()
DEFAULT_ACTUATOR = ActuatorConfig()
