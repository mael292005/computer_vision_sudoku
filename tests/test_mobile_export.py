"""Le CNN de l'application mobile doit rester identique au modèle ONNX du package."""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "training" / "export_mobile_weights.py"


def _export_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("export_mobile_weights", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mobile_weights_match_the_onnx_model() -> None:
    export = _export_module()
    expected = export.pack_weights(export.onnx_weights())
    assert export.WEIGHTS_PATH.read_bytes() == expected, (
        "mobile/assets/digits_cnn.bin est périmé : relancer training/export_mobile_weights.py"
    )


def test_mobile_weights_have_the_expected_size() -> None:
    export = _export_module()
    count = sum(int(np.prod(shape)) for _, shape in export.LAYERS)
    assert count == 55305
    assert export.WEIGHTS_PATH.stat().st_size == 4 * count


def test_reference_digits_cover_every_site() -> None:
    export = _export_module()
    cases = json.loads(export.REFERENCE_PATH.read_text(encoding="utf-8"))
    sources = {case["source"].split("_")[0] for case in cases}
    assert {"websudoku", "sudokucom", "onlinesudoku"} <= sources
    for case in cases:
        assert len(case["pixels"]) == 28 * 28
        assert int(np.argmax(case["logits"])) + 1 == case["expected_digit"]
