// Solveur de Sudoku : portage de src/sudoku_vision/solver.py.
//
// Backtracking guidé par l'heuristique MRV : on remplit d'abord la case qui a
// le moins de candidats. Les candidats de chaque ligne, colonne et bloc sont
// stockés en masques de bits (bit d = chiffre d déjà utilisé).

/// Grille 9x9, 0 = case vide.
typedef Grid = List<List<int>>;

const int gridSize = 9;
const int _box = 3;
const int _allDigits = 0x3FE; // bits 1 à 9

class InvalidGridError implements Exception {
  InvalidGridError(this.message, [this.cells = const []]);

  final String message;

  /// Cases en conflit, sous la forme (ligne, colonne).
  final List<(int, int)> cells;

  @override
  String toString() => 'InvalidGridError: $message';
}

class UnsolvableGridError implements Exception {
  UnsolvableGridError(this.message);

  final String message;

  @override
  String toString() => 'UnsolvableGridError: $message';
}

Grid emptyGrid() => List.generate(gridSize, (_) => List.filled(gridSize, 0));

Grid copyGrid(Grid grid) => [for (final row in grid) List<int>.of(row)];

int _boxIndex(int r, int c) => (r ~/ _box) * _box + c ~/ _box;

int _popcount(int mask) {
  var count = 0;
  while (mask != 0) {
    mask &= mask - 1;
    count++;
  }
  return count;
}

/// Cases où un même chiffre apparaît deux fois dans une ligne, colonne ou bloc.
List<(int, int)> findConflicts(Grid grid) {
  final conflicts = <(int, int)>{};
  final groups = <List<(int, int)>>[];
  for (var i = 0; i < gridSize; i++) {
    groups.add([for (var j = 0; j < gridSize; j++) (i, j)]);
    groups.add([for (var j = 0; j < gridSize; j++) (j, i)]);
    groups.add([
      for (var j = 0; j < gridSize; j++)
        ((i ~/ _box) * _box + j ~/ _box, (i % _box) * _box + j % _box),
    ]);
  }
  for (final group in groups) {
    final seen = <int, (int, int)>{};
    for (final cell in group) {
      final value = grid[cell.$1][cell.$2];
      if (value == 0) continue;
      final previous = seen[value];
      if (previous != null) {
        conflicts
          ..add(previous)
          ..add(cell);
      } else {
        seen[value] = cell;
      }
    }
  }
  return conflicts.toList();
}

/// Vérifie la forme de la grille et l'absence de doublons.
void validateGrid(Grid grid) {
  if (grid.length != gridSize || grid.any((row) => row.length != gridSize)) {
    throw InvalidGridError('la grille doit faire 9x9');
  }
  for (var r = 0; r < gridSize; r++) {
    for (var c = 0; c < gridSize; c++) {
      final value = grid[r][c];
      if (value < 0 || value > gridSize) {
        throw InvalidGridError('valeur invalide $value en ($r, $c)');
      }
    }
  }
  final conflicts = findConflicts(grid);
  if (conflicts.isNotEmpty) {
    throw InvalidGridError('un même chiffre apparaît deux fois', conflicts);
  }
}

/// Résout une grille ; la grille d'entrée n'est pas modifiée.
Grid solve(Grid grid) {
  validateGrid(grid);
  final board = copyGrid(grid);
  final rows = List.filled(gridSize, 0);
  final cols = List.filled(gridSize, 0);
  final boxes = List.filled(gridSize, 0);
  final empties = <(int, int)>[];

  for (var r = 0; r < gridSize; r++) {
    for (var c = 0; c < gridSize; c++) {
      final value = board[r][c];
      if (value != 0) {
        final bit = 1 << value;
        rows[r] |= bit;
        cols[c] |= bit;
        boxes[_boxIndex(r, c)] |= bit;
      } else {
        empties.add((r, c));
      }
    }
  }

  int candidates(int r, int c) =>
      _allDigits & ~(rows[r] | cols[c] | boxes[_boxIndex(r, c)]);

  bool backtrack() {
    if (empties.isEmpty) return true;

    // MRV : la case vide qui a le moins de candidats.
    var bestIndex = -1;
    var bestMask = 0;
    var bestCount = gridSize + 1;
    for (var i = 0; i < empties.length; i++) {
      final (r, c) = empties[i];
      final mask = candidates(r, c);
      final count = _popcount(mask);
      if (count < bestCount) {
        bestIndex = i;
        bestMask = mask;
        bestCount = count;
        if (count <= 1) break;
      }
    }
    if (bestCount == 0) return false;

    final (r, c) = empties.removeAt(bestIndex);
    final b = _boxIndex(r, c);
    var mask = bestMask;
    while (mask != 0) {
      final bit = mask & -mask;
      mask ^= bit;
      board[r][c] = bit.bitLength - 1;
      rows[r] |= bit;
      cols[c] |= bit;
      boxes[b] |= bit;
      if (backtrack()) return true;
      rows[r] ^= bit;
      cols[c] ^= bit;
      boxes[b] ^= bit;
    }
    board[r][c] = 0;
    empties.insert(bestIndex, (r, c));
    return false;
  }

  if (!backtrack()) {
    throw UnsolvableGridError("cette grille n'a pas de solution");
  }
  return board;
}

/// Grille depuis 81 caractères (chiffres, 0 ou . pour les cases vides).
Grid parseGrid(String text) {
  final cells = text.split('').where((ch) => !' \t\r\n|-+'.contains(ch)).toList();
  if (cells.length != gridSize * gridSize) {
    throw InvalidGridError('81 cases attendues, ${cells.length} trouvées');
  }
  final values = <int>[];
  for (final ch in cells) {
    if (ch == '.') {
      values.add(0);
    } else {
      final value = int.tryParse(ch);
      if (value == null) throw InvalidGridError('caractère inattendu « $ch »');
      values.add(value);
    }
  }
  return [
    for (var r = 0; r < gridSize; r++) values.sublist(r * gridSize, (r + 1) * gridSize),
  ];
}
