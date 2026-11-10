"""Le jeu de test lui-même est cohérent : fichiers présents, grilles valides et solubles."""

from sudoku_vision.solver import solve, validate_grid
from tests.samples import Sample


def test_image_file_exists(sample: Sample) -> None:
    assert sample.path.is_file()


def test_ground_truth_is_a_valid_puzzle(sample: Sample) -> None:
    validate_grid(sample.grid)
    solution = solve(sample.grid)
    assert all(0 not in row for row in solution)
