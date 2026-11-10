"""Solveur de Sudoku, totalement indépendant de la partie vision.

Algorithme : backtracking (essai-erreur avec retour en arrière) guidé par
l'heuristique MRV (*Minimum Remaining Values*) : à chaque étape, on remplit
d'abord la case qui a le moins de candidats possibles. Une case à un seul
candidat est donc remplie sans hésiter, et une case à zéro candidat révèle
une impasse immédiatement.

Les candidats de chaque ligne, colonne et bloc sont stockés sous forme de
masques de bits (bit ``d`` à 1 = chiffre ``d`` déjà utilisé), ce qui rend le
calcul des possibilités d'une case très rapide.

La grille est une liste de 9 listes de 9 entiers, ``0`` désignant une case vide.
"""

from __future__ import annotations

from collections.abc import Sequence

from sudoku_vision.config import BOX_SIZE, GRID_SIZE

Grid = list[list[int]]
"""Grille 9x9 d'entiers de 0 à 9, où 0 signifie « case vide »."""

_ALL_DIGITS = 0b11_1111_1110  # bits 1 à 9 à 1


class InvalidGridError(ValueError):
    """La grille est mal formée ou contient déjà un conflit (doublon)."""


class UnsolvableGridError(ValueError):
    """La grille est valide mais n'admet aucune solution."""


def _box_index(row: int, col: int) -> int:
    return (row // BOX_SIZE) * BOX_SIZE + col // BOX_SIZE


def validate_grid(grid: Sequence[Sequence[int]]) -> None:
    """Vérifie la forme de la grille et l'absence de doublons.

    Raises:
        InvalidGridError: si la grille n'est pas 9x9, contient une valeur hors
            de 0..9, ou un chiffre répété dans une ligne, une colonne ou un bloc.
    """
    if len(grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in grid):
        raise InvalidGridError(f"la grille doit faire {GRID_SIZE}x{GRID_SIZE}")

    rows = [0] * GRID_SIZE
    cols = [0] * GRID_SIZE
    boxes = [0] * GRID_SIZE
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            value = grid[r][c]
            if not isinstance(value, int) or not 0 <= value <= GRID_SIZE:
                raise InvalidGridError(f"valeur invalide {value!r} en ({r}, {c})")
            if value == 0:
                continue
            bit = 1 << value
            b = _box_index(r, c)
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                raise InvalidGridError(f"le chiffre {value} est en double autour de ({r}, {c})")
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit


def solve(grid: Sequence[Sequence[int]]) -> Grid:
    """Résout une grille de Sudoku.

    La grille d'entrée n'est pas modifiée : une nouvelle grille est renvoyée.

    Raises:
        InvalidGridError: si la grille est mal formée ou contient un conflit.
        UnsolvableGridError: si la grille n'a aucune solution.
    """
    validate_grid(grid)
    board = [list(row) for row in grid]

    rows = [0] * GRID_SIZE
    cols = [0] * GRID_SIZE
    boxes = [0] * GRID_SIZE
    empties: list[tuple[int, int]] = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            value = board[r][c]
            if value:
                bit = 1 << value
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box_index(r, c)] |= bit
            else:
                empties.append((r, c))

    def candidates(r: int, c: int) -> int:
        return _ALL_DIGITS & ~(rows[r] | cols[c] | boxes[_box_index(r, c)])

    def backtrack() -> bool:
        if not empties:
            return True

        # MRV : on choisit la case vide qui a le moins de candidats.
        best_i, best_mask, best_count = -1, 0, GRID_SIZE + 1
        for i, (r, c) in enumerate(empties):
            mask = candidates(r, c)
            count = mask.bit_count()
            if count < best_count:
                best_i, best_mask, best_count = i, mask, count
                if count <= 1:
                    break
        if best_count == 0:
            return False  # impasse : une case n'a plus aucun chiffre possible

        r, c = empties.pop(best_i)
        b = _box_index(r, c)
        mask = best_mask
        while mask:
            bit = mask & -mask  # bit de poids faible = plus petit candidat
            mask ^= bit
            board[r][c] = bit.bit_length() - 1
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if backtrack():
                return True
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit

        board[r][c] = 0
        empties.insert(best_i, (r, c))
        return False

    if not backtrack():
        raise UnsolvableGridError("cette grille n'a pas de solution")
    return board


def parse_grid(text: str) -> Grid:
    """Construit une grille depuis une chaîne de 81 caractères.

    Les chiffres 1-9 sont des cases remplies ; ``0`` ou ``.`` sont des cases
    vides. Les espaces, retours à la ligne et séparateurs ``|``, ``-``, ``+``
    sont ignorés, ce qui permet d'écrire la grille de façon lisible.

    Raises:
        InvalidGridError: si la chaîne ne contient pas exactement 81 cases.
    """
    cells = [ch for ch in text if ch not in " \t\r\n|-+"]
    if len(cells) != GRID_SIZE * GRID_SIZE:
        raise InvalidGridError(f"81 cases attendues, {len(cells)} trouvées")
    values: list[int] = []
    for ch in cells:
        if ch == ".":
            values.append(0)
        elif ch.isdigit():
            values.append(int(ch))
        else:
            raise InvalidGridError(f"caractère inattendu {ch!r}")
    return [values[i : i + GRID_SIZE] for i in range(0, len(values), GRID_SIZE)]


def format_grid(grid: Sequence[Sequence[int]]) -> str:
    """Affiche une grille de façon lisible, avec les blocs 3x3 séparés."""
    separator = "------+-------+------"
    lines: list[str] = []
    for r, row in enumerate(grid):
        if r and r % BOX_SIZE == 0:
            lines.append(separator)
        chunks = [
            " ".join(str(v) if v else "." for v in row[i : i + BOX_SIZE])
            for i in range(0, GRID_SIZE, BOX_SIZE)
        ]
        lines.append(" | ".join(chunks))
    return "\n".join(lines)
