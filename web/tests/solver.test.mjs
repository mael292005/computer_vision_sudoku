import assert from "node:assert/strict";
import { test } from "node:test";

import {
  InvalidGridError,
  UnsolvableGridError,
  emptyGrid,
  findConflicts,
  parseGrid,
  solve,
} from "../js/solver.js";

const EASY =
  "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
const EASY_SOLUTION =
  "534678912672195348198342567859761423426853791713924856961537284287419635345286179";
const HARD =
  "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..";

function isValidSolution(grid) {
  const full = (values) => new Set(values).size === 9 && !values.includes(0);
  for (let i = 0; i < 9; i += 1) {
    const row = grid[i];
    const col = grid.map((r) => r[i]);
    const br = Math.floor(i / 3) * 3;
    const bc = (i % 3) * 3;
    const box = [0, 1, 2].flatMap((dr) => [0, 1, 2].map((dc) => grid[br + dr][bc + dc]));
    if (!full(row) || !full(col) || !full(box)) return false;
  }
  return true;
}

test("résout une grille facile", () => {
  assert.deepEqual(solve(parseGrid(EASY)), parseGrid(EASY_SOLUTION));
});

test("résout une grille très difficile rapidement", () => {
  const start = performance.now();
  const solution = solve(parseGrid(HARD));
  assert.ok(isValidSolution(solution));
  assert.ok(performance.now() - start < 1000);
});

test("ne modifie pas la grille d'entrée", () => {
  const puzzle = parseGrid(EASY);
  const copy = puzzle.map((row) => [...row]);
  solve(puzzle);
  assert.deepEqual(puzzle, copy);
});

test("résout une grille vide", () => {
  assert.ok(isValidSolution(solve(emptyGrid())));
});

test("signale les cases en conflit", () => {
  const grid = parseGrid(EASY);
  grid[0][2] = 5; // deux 5 sur la première ligne
  assert.throws(() => solve(grid), (error) => {
    assert.ok(error instanceof InvalidGridError);
    const cells = error.cells.map(([r, c]) => `${r},${c}`).sort();
    assert.deepEqual(cells, ["0,0", "0,2"]);
    return true;
  });
  assert.equal(findConflicts(parseGrid(EASY)).length, 0);
});

test("détecte une grille sans solution", () => {
  const grid = emptyGrid();
  grid[0] = [1, 2, 3, 4, 5, 6, 7, 8, 0];
  grid[1][8] = 9;
  assert.throws(() => solve(grid), UnsolvableGridError);
});

test("rejette les grilles mal formées", () => {
  assert.throws(() => solve([[0]]), InvalidGridError);
  const bad = emptyGrid();
  bad[0][0] = 10;
  assert.throws(() => solve(bad), InvalidGridError);
  assert.throws(() => parseGrid("123"), InvalidGridError);
  assert.throws(() => parseGrid("x".repeat(81)), InvalidGridError);
});
