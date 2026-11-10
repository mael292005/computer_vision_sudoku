"""Saisie automatique de la solution : clic sur chaque case vide, puis frappe du chiffre.

La partie « souris et clavier » est isolée derrière une petite interface
(:class:`Controller`). Le vrai contrôleur utilise ``pyautogui`` ; les tests
utilisent un faux contrôleur qui se contente d'enregistrer les actions.

Sécurité : avec ``pyautogui``, amener la souris dans un coin de l'écran
arrête immédiatement le bot (*failsafe*).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from sudoku_vision.config import DEFAULT_ACTUATOR, GRID_SIZE, ActuatorConfig


class Controller(Protocol):
    """Ce dont le bot a besoin pour agir sur l'écran."""

    def click(self, x: float, y: float) -> None: ...

    def type_digit(self, digit: int) -> None: ...


class PyAutoGuiController:  # pragma: no cover - nécessite un écran
    """Contrôleur réel, basé sur ``pyautogui``."""

    def __init__(self, config: ActuatorConfig = DEFAULT_ACTUATOR) -> None:
        import pyautogui

        pyautogui.PAUSE = config.pause_seconds
        pyautogui.FAILSAFE = config.failsafe
        self._gui = pyautogui

    def click(self, x: float, y: float) -> None:
        self._gui.click(round(x), round(y))

    def type_digit(self, digit: int) -> None:
        self._gui.press(str(digit))


def fill_grid(
    puzzle: Sequence[Sequence[int]],
    solution: Sequence[Sequence[int]],
    centers: npt.NDArray[np.float32],
    controller: Controller,
) -> int:
    """Remplit les cases vides du puzzle avec la solution.

    Args:
        puzzle: grille de départ (0 = case vide), seules ces cases sont remplies.
        solution: grille résolue.
        centers: centre de chaque case en coordonnées souris, tableau (9, 9, 2).
        controller: qui réalise les clics et les frappes.

    Returns:
        Le nombre de cases remplies.
    """
    if np.asarray(centers).shape != (GRID_SIZE, GRID_SIZE, 2):
        raise ValueError("centers doit être un tableau (9, 9, 2)")
    filled = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if puzzle[r][c] == 0:
                x, y = centers[r, c]
                controller.click(float(x), float(y))
                controller.type_digit(solution[r][c])
                filled += 1
    return filled
