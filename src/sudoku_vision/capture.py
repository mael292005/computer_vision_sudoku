"""Capture de l'écran et conversion des coordonnées image vers l'écran.

Piège classique : sur un écran à mise à l'échelle (Windows à 125 % ou 150 %,
Mac Retina), la capture est en pixels physiques alors que la souris peut
travailler en pixels « logiques ». On mesure donc le rapport entre les deux
pour cliquer au bon endroit.

Nécessite l'option ``screen`` (``mss`` et ``pyautogui``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sudoku_vision.grid import Image


@dataclass(frozen=True)
class ScreenShot:
    """Une capture d'écran et de quoi convertir ses pixels en coordonnées souris."""

    image: Image
    """Image BGR de l'écran."""

    left: int = 0
    """Position de l'écran capturé sur le bureau, en pixels physiques."""

    top: int = 0

    scale: float = 1.0
    """Pixels souris par pixel de capture (0.5 sur un écran Retina, par exemple)."""

    def to_screen(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convertit des coordonnées (x, y) de l'image en coordonnées souris."""
        offset = np.array([self.left, self.top], dtype=np.float32)
        return np.asarray((points + offset) * self.scale, dtype=np.float32)


def grab_screen(monitor: int = 1) -> ScreenShot:  # pragma: no cover - nécessite un écran
    """Capture un écran (1 = écran principal).

    Raises:
        ImportError: si l'option ``screen`` n'est pas installée.
    """
    import mss
    import pyautogui

    with mss.mss() as sct:
        area = sct.monitors[monitor]
        raw = np.asarray(sct.grab(area), dtype=np.uint8)
    image = np.ascontiguousarray(raw[:, :, :3])  # BGRA -> BGR
    logical_width = pyautogui.size().width
    scale = logical_width / area["width"] if monitor == 1 else 1.0
    return ScreenShot(image=image, left=area["left"], top=area["top"], scale=scale)
