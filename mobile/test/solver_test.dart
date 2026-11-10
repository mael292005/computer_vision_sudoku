import 'package:flutter_test/flutter_test.dart';
import 'package:sudoku_photo/core/solver.dart';

const easy = '530070000600195000098000060800060003400803001700020006060000280000419005000080079';
const easySolution =
    '534678912672195348198342567859761423426853791713924856961537284287419635345286179';
const hard = '8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..';

bool isValidSolution(Grid grid) {
  bool full(Iterable<int> values) => values.toSet().length == 9 && !values.contains(0);
  for (var i = 0; i < 9; i++) {
    final br = (i ~/ 3) * 3;
    final bc = (i % 3) * 3;
    if (!full(grid[i]) ||
        !full([for (final row in grid) row[i]]) ||
        !full([for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) grid[br + r][bc + c]])) {
      return false;
    }
  }
  return true;
}

void main() {
  test('résout une grille facile', () {
    expect(solve(parseGrid(easy)), parseGrid(easySolution));
  });

  test('résout une grille très difficile rapidement', () {
    final watch = Stopwatch()..start();
    expect(isValidSolution(solve(parseGrid(hard))), isTrue);
    expect(watch.elapsedMilliseconds, lessThan(2000));
  });

  test("ne modifie pas la grille d'entrée", () {
    final puzzle = parseGrid(easy);
    final copy = copyGrid(puzzle);
    solve(puzzle);
    expect(puzzle, copy);
  });

  test('résout une grille vide', () {
    expect(isValidSolution(solve(emptyGrid())), isTrue);
  });

  test('signale les cases en conflit', () {
    final grid = parseGrid(easy);
    grid[0][2] = 5; // deux 5 sur la première ligne
    expect(
      () => solve(grid),
      throwsA(isA<InvalidGridError>().having((e) => e.cells.toSet(), 'cells', {(0, 0), (0, 2)})),
    );
    expect(findConflicts(parseGrid(easy)), isEmpty);
  });

  test('détecte une grille sans solution', () {
    final grid = emptyGrid();
    grid[0] = [1, 2, 3, 4, 5, 6, 7, 8, 0];
    grid[1][8] = 9;
    expect(() => solve(grid), throwsA(isA<UnsolvableGridError>()));
  });

  test('rejette les grilles mal formées', () {
    expect(() => solve([
          [0],
        ]), throwsA(isA<InvalidGridError>()));
    expect(() => parseGrid('123'), throwsA(isA<InvalidGridError>()));
    expect(() => parseGrid('x' * 81), throwsA(isA<InvalidGridError>()));
  });
}
