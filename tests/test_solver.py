"""Tests du solveur : grilles faciles, difficiles, invalides et insolubles."""

import time

import pytest

from sudoku_vision.solver import (
    InvalidGridError,
    UnsolvableGridError,
    format_grid,
    parse_grid,
    solve,
    validate_grid,
)

EASY = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
EASY_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"

# Grille connue pour être très difficile pour un backtracking naïf.
HARD = "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."


def _is_valid_solution(grid: list[list[int]]) -> bool:
    expected = set(range(1, 10))
    rows = all(set(row) == expected for row in grid)
    cols = all({grid[r][c] for r in range(9)} == expected for c in range(9))
    boxes = all(
        {grid[br + r][bc + c] for r in range(3) for c in range(3)} == expected
        for br in (0, 3, 6)
        for bc in (0, 3, 6)
    )
    return rows and cols and boxes


def test_solves_easy_grid() -> None:
    assert solve(parse_grid(EASY)) == parse_grid(EASY_SOLUTION)


def test_solves_hard_grid_quickly() -> None:
    start = time.perf_counter()
    solution = solve(parse_grid(HARD))
    elapsed = time.perf_counter() - start
    assert _is_valid_solution(solution)
    assert elapsed < 1.0, f"résolution trop lente : {elapsed:.2f} s"


def test_solution_keeps_the_given_digits() -> None:
    puzzle = parse_grid(HARD)
    solution = solve(puzzle)
    for r in range(9):
        for c in range(9):
            if puzzle[r][c]:
                assert solution[r][c] == puzzle[r][c]


def test_input_grid_is_not_modified() -> None:
    puzzle = parse_grid(EASY)
    copy = [row[:] for row in puzzle]
    solve(puzzle)
    assert puzzle == copy


def test_empty_grid_has_a_solution() -> None:
    assert _is_valid_solution(solve([[0] * 9 for _ in range(9)]))


def test_already_solved_grid_is_returned_as_is() -> None:
    solved = parse_grid(EASY_SOLUTION)
    assert solve(solved) == solved


def test_duplicate_in_row_is_rejected() -> None:
    grid = parse_grid(EASY)
    grid[0][2] = 5  # deux 5 sur la première ligne
    with pytest.raises(InvalidGridError):
        solve(grid)


def test_duplicate_in_box_is_rejected() -> None:
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 7
    grid[2][2] = 7
    with pytest.raises(InvalidGridError):
        validate_grid(grid)


@pytest.mark.parametrize(
    "grid",
    [
        [[0] * 9 for _ in range(8)],  # 8 lignes
        [[0] * 8 for _ in range(9)],  # 8 colonnes
        [[10] + [0] * 8] + [[0] * 9 for _ in range(8)],  # valeur hors limites
    ],
)
def test_malformed_grid_is_rejected(grid: list[list[int]]) -> None:
    with pytest.raises(InvalidGridError):
        validate_grid(grid)


def test_unsolvable_grid_is_detected() -> None:
    # Aucun doublon, mais la case (0, 8) ne peut recevoir aucun chiffre :
    # 1 à 8 sont sur sa ligne et 9 est dans sa colonne.
    grid = [[0] * 9 for _ in range(9)]
    grid[0][:8] = [1, 2, 3, 4, 5, 6, 7, 8]
    grid[1][8] = 9
    with pytest.raises(UnsolvableGridError):
        solve(grid)


def test_parse_grid_accepts_readable_layout() -> None:
    text = format_grid(parse_grid(EASY))
    assert parse_grid(text) == parse_grid(EASY)


def test_parse_grid_rejects_wrong_length() -> None:
    with pytest.raises(InvalidGridError):
        parse_grid("123")


def test_parse_grid_rejects_unknown_characters() -> None:
    with pytest.raises(InvalidGridError):
        parse_grid("x" * 81)


def test_format_grid_layout() -> None:
    lines = format_grid(parse_grid(EASY)).splitlines()
    assert len(lines) == 11  # 9 lignes + 2 séparateurs
    assert lines[0] == "5 3 . | . 7 . | . . ."
