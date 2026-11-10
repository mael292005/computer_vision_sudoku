"""Fixtures partagées par les tests."""

from __future__ import annotations

import pytest

from tests.samples import SAMPLES, Sample


@pytest.fixture(params=SAMPLES, ids=[s.path.stem for s in SAMPLES])
def sample(request: pytest.FixtureRequest) -> Sample:
    """Paramètre un test sur chaque capture du jeu de test."""
    result: Sample = request.param
    return result
