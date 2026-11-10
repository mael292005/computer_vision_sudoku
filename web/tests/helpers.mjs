// Outils communs aux tests : chargement d'OpenCV.js, du CNN et du jeu de test.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import cvModule from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";

import { parseGrid } from "../js/solver.js";

const here = dirname(fileURLToPath(import.meta.url));
export const ROOT = join(here, "..", "..");
export const DATA_DIR = join(ROOT, "tests", "data");
export const MODEL_PATH = join(ROOT, "src", "sudoku_vision", "assets", "digits_cnn.onnx");

ort.env.wasm.numThreads = 1;
export { ort };

let cvPromise;
/**
 * OpenCV.js, une fois son module WebAssembly prêt.
 *
 * Le module est lui-même un « thenable » : on le renvoie emballé dans un objet,
 * sinon `await` tenterait de le résoudre à nouveau.
 */
export async function loadCv() {
  cvPromise ??= new Promise((resolve) => {
    if (cvModule.Mat) resolve({ cv: cvModule });
    else cvModule.then((ready) => resolve({ cv: ready }));
  });
  return (await cvPromise).cv;
}

/** Les captures du jeu de test et leur grille attendue (même source que les tests Python). */
export function loadSamples() {
  const raw = JSON.parse(readFileSync(join(DATA_DIR, "grids.json"), "utf8"));
  return Object.entries(raw)
    .filter(([name]) => !name.startsWith("_"))
    .map(([name, entry]) => ({
      name,
      level: entry.level,
      path: join(DATA_DIR, name),
      grid: parseGrid(entry.grid.join("")),
    }));
}

/** Charge un PNG ou un JPEG en Mat RGBA, comme cv.imread sur un canvas dans le navigateur. */
export function readPng(cv, path) {
  const bytes = readFileSync(path);
  const image = /\.jpe?g$/i.test(path)
    ? jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true })
    : PNG.sync.read(bytes);
  return cv.matFromImageData({ data: image.data, width: image.width, height: image.height });
}
