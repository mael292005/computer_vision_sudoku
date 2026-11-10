// Copie dans web/assets/ les fichiers partagés avec la version Python :
// le CNN (ONNX) et les captures d'exemple. Une seule source de vérité,
// donc rien de dupliqué dans le dépôt.

import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const root = join(web, "..");
const out = join(web, "assets");
mkdirSync(join(out, "examples"), { recursive: true });

const files = [
  ["src/sudoku_vision/assets/digits_cnn.onnx", "digits_cnn.onnx"],
  ["tests/data/websudoku_easy.png", "examples/easy.png"],
  ["tests/data/websudoku_medium.png", "examples/medium.png"],
  ["tests/data/websudoku_hard.png", "examples/hard.png"],
  ["tests/data/websudoku_evil.png", "examples/evil.png"],
];
for (const [from, to] of files) {
  copyFileSync(join(root, from), join(out, to));
  console.log(`${from} -> web/assets/${to}`);
}
