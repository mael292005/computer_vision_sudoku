// Interface du site : saisie de la grille, lecture d'image, affichage de la solution.

import { InvalidGridError, UnsolvableGridError, emptyGrid, solve } from "./solver.js";
import { GridNotFoundError, loadModel, readGrid } from "./vision.js";

// Versions identiques à web/package.json (celles utilisées par les tests).
const OPENCV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js";
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
const MODEL_URL = "assets/digits_cnn.onnx";
const MAX_IMAGE_SIDE = 2600;
const DOUBT_THRESHOLD = 0.9;

const boardEl = document.getElementById("board");
const frameEl = boardEl.parentElement;
const statusEl = document.getElementById("status");
const engineEl = document.getElementById("engine");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file");
const insightEl = document.getElementById("insight");
const insightText = document.getElementById("insight-text");
const solveButton = document.getElementById("solve");

/** @type {HTMLInputElement[][]} */
const cells = [];

// --- Grille -----------------------------------------------------------------

function buildBoard() {
  for (let r = 0; r < 9; r += 1) {
    const row = [];
    for (let c = 0; c < 9; c += 1) {
      const input = document.createElement("input");
      input.className = "cell";
      input.inputMode = "numeric";
      input.maxLength = 1;
      input.autocomplete = "off";
      input.setAttribute("aria-label", `Ligne ${r + 1}, colonne ${c + 1}`);
      input.style.setProperty("--row", String(r));
      input.dataset.r = String(r);
      input.dataset.c = String(c);
      input.addEventListener("input", onCellInput);
      input.addEventListener("keydown", onCellKey);
      input.addEventListener("focus", () => input.select());
      boardEl.append(input);
      row.push(input);
    }
    cells.push(row);
  }
}

function onCellInput(event) {
  const input = event.target;
  const digit = input.value.replace(/[^1-9]/g, "").slice(-1);
  input.value = digit;
  input.classList.remove("doubtful");
  clearSolution();
  clearMarks();
  setStatus("");
  if (digit) move(input, 0, 1);
}

function onCellKey(event) {
  const input = event.target;
  const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (moves[event.key]) {
    event.preventDefault();
    move(input, ...moves[event.key]);
  } else if (event.key === "Enter") {
    solveCurrent();
  } else if ((event.key === "Backspace" || event.key === "Delete") && !input.value) {
    event.preventDefault();
    move(input, 0, -1);
  }
}

function move(input, dr, dc) {
  let index = Number(input.dataset.r) * 9 + Number(input.dataset.c) + dr * 9 + dc;
  index = Math.min(80, Math.max(0, index));
  cells[Math.floor(index / 9)][index % 9].focus();
}

/** Chiffres de départ : tout ce qui n'a pas été trouvé par le solveur. */
function readPuzzle() {
  return cells.map((row) =>
    row.map((input) => (input.classList.contains("found") ? 0 : Number(input.value) || 0)),
  );
}

function fillPuzzle(grid, confidence) {
  clearMarks();
  cells.forEach((row, r) =>
    row.forEach((input, c) => {
      input.classList.remove("found");
      input.value = grid[r][c] ? String(grid[r][c]) : "";
      input.classList.toggle("doubtful", Boolean(confidence && grid[r][c] && confidence[r][c] < DOUBT_THRESHOLD));
    }),
  );
}

function showSolution(puzzle, solution) {
  frameEl.classList.remove("scanning");
  void frameEl.offsetWidth; // relance l'animation
  frameEl.classList.add("scanning");
  cells.forEach((row, r) =>
    row.forEach((input, c) => {
      if (!puzzle[r][c]) {
        input.value = String(solution[r][c]);
        input.classList.add("found");
      }
    }),
  );
}

function clearSolution() {
  for (const input of cells.flat()) {
    if (input.classList.contains("found")) {
      input.value = "";
      input.classList.remove("found");
    }
  }
}

function clearMarks() {
  for (const input of cells.flat()) input.classList.remove("conflict");
}

function setStatus(message, kind = "ok") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`;
}

/** Résout la grille affichée ; renvoie le nombre de cases remplies, ou null en cas d'échec. */
function solveCurrent() {
  clearSolution();
  clearMarks();
  const puzzle = readPuzzle();
  const givens = puzzle.flat().filter(Boolean).length;
  if (givens === 0) {
    setStatus("La grille est vide : tapez des chiffres ou envoyez une capture.", "error");
    return null;
  }
  try {
    const solution = solve(puzzle);
    showSolution(puzzle, solution);
    const filled = 81 - givens;
    setStatus(`Grille résolue : ${filled} cases remplies.`);
    return filled;
  } catch (error) {
    if (error instanceof InvalidGridError) {
      for (const [r, c] of error.cells) cells[r][c].classList.add("conflict");
      setStatus("Un même chiffre apparaît deux fois (en rouge). Corrigez-le puis cliquez sur Résoudre.", "error");
    } else if (error instanceof UnsolvableGridError) {
      setStatus("Cette grille n'a pas de solution. Vérifiez les chiffres saisis.", "error");
    } else {
      throw error;
    }
    return null;
  }
}

// --- Moteur de vision ---------------------------------------------------------

let enginePromise;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`impossible de charger ${src}`));
    document.head.append(script);
  });
}

/** Charge OpenCV.js, ONNX Runtime et le CNN, une seule fois. */
function loadEngine() {
  enginePromise ??= (async () => {
    engineEl.textContent = "Chargement du moteur de vision (environ 15 Mo, une seule fois)…";
    const ortPromise = import(`${ORT_BASE}ort.min.mjs`);
    await loadScript(OPENCV_URL);
    // OpenCV.js est un « thenable » : on l'emballe pour que await ne le résolve pas à nouveau.
    const { cv } = await new Promise((resolve) => {
      const module = window.cv;
      if (module.Mat) resolve({ cv: module });
      else module.then((ready) => resolve({ cv: ready }));
    });
    const ort = await ortPromise;
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1;
    const session = await loadModel(ort, MODEL_URL);
    engineEl.textContent = "Moteur de vision prêt : OpenCV.js et le réseau de neurones tournent dans votre navigateur.";
    return { cv, ort, session };
  })();
  enginePromise.catch(() => {
    enginePromise = undefined;
    engineEl.textContent = "Le moteur de vision n'a pas pu se charger. Vérifiez votre connexion puis réessayez.";
  });
  return enginePromise;
}

// --- Lecture d'image ------------------------------------------------------------

function imageToCanvas(image) {
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(image.naturalWidth, image.naturalHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.naturalWidth * scale);
  canvas.height = Math.round(image.naturalHeight * scale);
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("image illisible"));
    image.src = url;
  });
}

function drawInsight(source, result) {
  const { corners, warped, size, filled } = result;

  // Capture recadrée autour de la grille, contour détecté en cobalt.
  const xs = corners.map((p) => p[0]);
  const ys = corners.map((p) => p[1]);
  const pad = Math.round((Math.max(...xs) - Math.min(...xs)) * 0.08);
  const x0 = Math.max(0, Math.min(...xs) - pad);
  const y0 = Math.max(0, Math.min(...ys) - pad);
  const x1 = Math.min(source.width, Math.max(...xs) + pad);
  const y1 = Math.min(source.height, Math.max(...ys) + pad);
  const detected = document.getElementById("detected");
  const scale = 360 / Math.max(x1 - x0, y1 - y0);
  detected.width = Math.round((x1 - x0) * scale);
  detected.height = Math.round((y1 - y0) * scale);
  const dctx = detected.getContext("2d");
  dctx.drawImage(source, x0, y0, x1 - x0, y1 - y0, 0, 0, detected.width, detected.height);
  dctx.strokeStyle = "#2445D4";
  dctx.lineWidth = 4;
  dctx.beginPath();
  corners.forEach(([x, y], i) => {
    const px = (x - x0) * scale;
    const py = (y - y0) * scale;
    if (i === 0) dctx.moveTo(px, py);
    else dctx.lineTo(px, py);
  });
  dctx.closePath();
  dctx.stroke();
  dctx.fillStyle = "#2445D4";
  for (const [x, y] of corners) {
    dctx.beginPath();
    dctx.arc((x - x0) * scale, (y - y0) * scale, 7, 0, Math.PI * 2);
    dctx.fill();
  }

  // Grille redressée, cases pleines teintées.
  const warpedCanvas = document.getElementById("warped");
  warpedCanvas.width = size;
  warpedCanvas.height = size;
  const wctx = warpedCanvas.getContext("2d");
  const pixels = wctx.createImageData(size, size);
  for (let i = 0; i < warped.length; i += 1) {
    pixels.data[i * 4] = warped[i];
    pixels.data[i * 4 + 1] = warped[i];
    pixels.data[i * 4 + 2] = warped[i];
    pixels.data[i * 4 + 3] = 255;
  }
  wctx.putImageData(pixels, 0, 0);
  const step = size / 9;
  wctx.fillStyle = "rgba(36, 69, 212, 0.16)";
  filled.forEach((row, r) =>
    row.forEach((isFilled, c) => {
      if (isFilled) wctx.fillRect(c * step + 2, r * step + 2, step - 4, step - 4);
    }),
  );
  insightEl.hidden = false;
}

async function readImage(url) {
  setBusy(true);
  setStatus("Lecture de l'image…");
  try {
    const [engine, image] = await Promise.all([loadEngine(), loadImage(url)]);
    const canvas = imageToCanvas(image);
    const rgba = engine.cv.imread(canvas);
    let result;
    try {
      result = await readGrid(engine.cv, engine.ort, engine.session, rgba);
    } finally {
      rgba.delete();
    }
    fillPuzzle(result.grid, result.confidence);
    drawInsight(canvas, result);

    const start = performance.now();
    const filled = solveCurrent();
    const t = result.timings;
    const read = result.grid.flat().filter(Boolean).length;
    const doubtful = result.confidence.flat().filter((v) => v < DOUBT_THRESHOLD).length;
    insightText.textContent =
      `Grille trouvée en ${Math.round(t.grille)} ms, ${read} chiffres lus en ` +
      `${Math.round(t.cases + t.chiffres)} ms, solution en ${Math.max(1, Math.round(performance.now() - start))} ms.` +
      (doubtful ? " Les chiffres soulignés en pointillés ont été lus avec moins de certitude : vérifiez-les." : "");
    if (filled !== null) {
      const total = Math.round(t.grille + t.cases + t.chiffres);
      setStatus(`${read} chiffres lus et grille résolue en ${total} ms : ${filled} cases remplies.`);
    }
  } catch (error) {
    if (error instanceof GridNotFoundError) {
      setStatus(
        "Aucune grille trouvée dans l'image. Essayez une capture où la grille est entière, nette et pas trop petite.",
        "error",
      );
    } else {
      setStatus(`L'image n'a pas pu être lue (${error.message}).`, "error");
    }
  } finally {
    setBusy(false);
  }
}

function readFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Ce fichier n'est pas une image. Envoyez une capture au format PNG ou JPEG.", "error");
    return;
  }
  const url = URL.createObjectURL(file);
  readImage(url).finally(() => URL.revokeObjectURL(url));
}

function setBusy(busy) {
  solveButton.disabled = busy;
  for (const button of document.querySelectorAll("[data-example]")) button.disabled = busy;
  document.body.style.cursor = busy ? "progress" : "";
}

// --- Branchements ------------------------------------------------------------------

buildBoard();

solveButton.addEventListener("click", solveCurrent);
document.getElementById("clear").addEventListener("click", () => {
  fillPuzzle(emptyGrid());
  insightEl.hidden = true;
  setStatus("");
  cells[0][0].focus();
});

fileInput.addEventListener("change", () => {
  readFile(fileInput.files[0]);
  fileInput.value = "";
});

for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
}
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  readFile(event.dataTransfer.files[0]);
});

document.addEventListener("paste", (event) => {
  const item = [...event.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) {
    event.preventDefault();
    readFile(item.getAsFile());
  }
});

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => readImage(`assets/examples/${button.dataset.example}.png`));
}

// Le moteur se charge en arrière-plan : il est prêt quand on en a besoin.
const warmUp = () => loadEngine().catch(() => {});
if ("requestIdleCallback" in window) requestIdleCallback(warmUp, { timeout: 2000 });
else setTimeout(warmUp, 500);
