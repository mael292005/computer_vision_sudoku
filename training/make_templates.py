"""Construit les modèles du template matching et les enregistre dans le package.

Usage :
    python training/make_templates.py
"""

from __future__ import annotations

import numpy as np

from sudoku_vision.digits import TEMPLATES_PATH
from sudoku_vision.synthetic import TEMPLATE_FONT_CANDIDATES, available_fonts, make_templates


def main() -> None:
    fonts = available_fonts(TEMPLATE_FONT_CANDIDATES)
    if not fonts:
        raise SystemExit("Aucune police Times trouvée (Liberation Serif ou Times New Roman).")
    templates = make_templates(fonts[0])
    TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TEMPLATES_PATH, templates=templates)
    print(f"Modèles construits avec {fonts[0].name} -> {TEMPLATES_PATH}")


if __name__ == "__main__":
    main()
