// Parité avec la version Python : le site doit lire les captures du jeu de
// test exactement comme le bot (tests/data/grids.json).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { before, test } from "node:test";

import { solve } from "../js/solver.js";
import {
  GridNotFoundError,
  detectGrid,
  extractDigit,
  loadModel,
  orderCorners,
  pyRound,
  readGrid,
} from "../js/vision.js";
import { MODEL_PATH, loadCv, loadSamples, ort, readPng } from "./helpers.mjs";

let cv;
let session;
const samples = loadSamples();

before(async () => {
  cv = await loadCv();
  session = await loadModel(ort, new Uint8Array(readFileSync(MODEL_PATH)));
});

test("pyRound arrondit les ,5 vers le pair, comme Python", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 2.4, 2.6, 12.5].map(pyRound), [0, 2, 2, 2, 3, 12]);
});

test("orderCorners range les coins dans le bon ordre", () => {
  const ordered = orderCorners([
    [100, 100],
    [0, 0],
    [0, 100],
    [100, 0],
  ]);
  assert.deepEqual(ordered, [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
  ]);
});

for (const sample of samples) {
  test(`lit la capture ${sample.level} comme la version Python`, async () => {
    const rgba = readPng(cv, sample.path);
    try {
      const result = await readGrid(cv, ort, session, rgba);
      assert.deepEqual(result.grid, sample.grid);
      const expectedFilled = sample.grid.map((row) => row.map((v) => v > 0));
      assert.deepEqual(result.filled, expectedFilled);
      assert.ok(solve(result.grid));
      const minConfidence = Math.min(...result.confidence.flat());
      assert.ok(minConfidence > 0.9, `confiance minimale ${minConfidence}`);
    } finally {
      rgba.delete();
    }
  });
}

test("une image sans grille lève GridNotFoundError", () => {
  const blank = new cv.Mat(300, 300, cv.CV_8UC4, new cv.Scalar(255, 255, 255, 255));
  try {
    assert.throws(() => detectGrid(cv, blank), GridNotFoundError);
  } finally {
    blank.delete();
  }
});

test("une case vide ne donne aucun chiffre", () => {
  const cell = { data: new Uint8Array(38 * 38).fill(255), width: 38, height: 38 };
  assert.equal(extractDigit(cv, cell), null);
});
