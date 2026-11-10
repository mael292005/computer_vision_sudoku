"""Exporte le CNN pour l'application mobile (réseau recodé en Dart pur).

Deux fichiers sont écrits dans mobile/ :

- ``assets/digits_cnn.bin`` : les 55 305 poids en float32 little-endian, dans
  l'ordre de :data:`LAYERS` (celui que lit mobile/lib/core/cnn.dart) ;
- ``test/fixtures/cnn_reference.json`` : quelques vrais chiffres extraits des
  captures et les sorties calculées par le modèle ONNX, pour vérifier que le
  réseau Dart donne les mêmes résultats.

Usage :
    python training/export_mobile_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from onnx import load, numpy_helper

from sudoku_vision.cells import split_cells
from sudoku_vision.digits import CNN_PATH, CnnClassifier, extract_digit
from sudoku_vision.evaluate import load_annotations
from sudoku_vision.grid import detect_grid, load_image

ROOT = Path(__file__).resolve().parent.parent
MOBILE = ROOT / "mobile"
WEIGHTS_PATH = MOBILE / "assets" / "digits_cnn.bin"
REFERENCE_PATH = MOBILE / "test" / "fixtures" / "cnn_reference.json"

LAYERS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("features.0.weight", (16, 1, 3, 3)),
    ("features.0.bias", (16,)),
    ("features.3.weight", (32, 16, 3, 3)),
    ("features.3.bias", (32,)),
    ("classifier.1.weight", (32, 1568)),
    ("classifier.1.bias", (32,)),
    ("classifier.4.weight", (9, 32)),
    ("classifier.4.bias", (9,)),
)


def onnx_weights(path: Path = CNN_PATH) -> dict[str, np.ndarray]:
    """Poids du modèle ONNX, par nom de couche."""
    model = load(str(path))
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def pack_weights(weights: dict[str, np.ndarray]) -> bytes:
    """Concatène les poids dans l'ordre attendu par le code Dart."""
    chunks = []
    for name, shape in LAYERS:
        array = weights[name]
        if array.shape != shape:
            raise ValueError(f"{name} : forme {array.shape}, attendu {shape}")
        chunks.append(np.ascontiguousarray(array, dtype="<f4").tobytes())
    return b"".join(chunks)


def reference_digits(per_sample: int = 2) -> list[dict[str, object]]:
    """Quelques chiffres réels de chaque capture, avec les sorties du modèle ONNX."""
    cnn = CnnClassifier()
    cases: list[dict[str, object]] = []
    for sample in load_annotations(ROOT / "tests" / "data"):
        cells = split_cells(detect_grid(load_image(sample.path)).warped)
        kept = 0
        for r in range(9):
            for c in range(9):
                if kept == per_sample or not sample.grid[r][c]:
                    continue
                digit = extract_digit(cells[r][c])
                if digit is None:
                    continue
                cnn._net.setInput(digit.astype(np.float32)[None, None] / 255.0)
                logits = cnn._net.forward().reshape(-1)
                cases.append(
                    {
                        "source": f"{sample.path.name} ligne {r + 1} colonne {c + 1}",
                        "expected_digit": sample.grid[r][c],
                        "pixels": digit.reshape(-1).tolist(),
                        "logits": [round(float(v), 6) for v in logits],
                    }
                )
                kept += 1
    return cases


def main() -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_bytes(pack_weights(onnx_weights()))
    print(f"Poids -> {WEIGHTS_PATH.relative_to(ROOT)} ({WEIGHTS_PATH.stat().st_size} octets)")

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = reference_digits()
    REFERENCE_PATH.write_text(json.dumps(cases) + "\n", encoding="utf-8")
    print(f"Référence -> {REFERENCE_PATH.relative_to(ROOT)} ({len(cases)} chiffres)")


if __name__ == "__main__":
    main()
