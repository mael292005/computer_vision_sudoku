// Pipeline de vision : portage de grid.py, cells.py et digits.py.
//
// Mêmes étapes et mêmes réglages que la version Python, pour que le site lise
// les grilles exactement comme le bot. Les tests (web/tests) le vérifient sur
// les captures du jeu de test.
//
// Le module ne charge rien lui-même : on lui passe `cv` (OpenCV.js) et `ort`
// (ONNX Runtime Web), ce qui permet de l'utiliser dans le navigateur comme
// dans Node pour les tests.

// Réglages : identiques à src/sudoku_vision/config.py.
export const CONFIG = Object.freeze({
  gridSize: 9,
  warpedSize: 450,
  cellMarginRatio: 0.12,
  emptyCellInkRatio: 0.01,
  digitSize: 28,
  digitBox: 20,
  minContrast: 60,
  minGridAreaRatio: 0.01,
  maxAspectError: 0.2,
});

export class GridNotFoundError extends Error {
  constructor(message = "aucune grille de Sudoku trouvée dans l'image") {
    super(message);
    this.name = "GridNotFoundError";
  }
}

/** Arrondi comme Python 3 : les ,5 vont vers l'entier pair. */
export function pyRound(x) {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Libère des Mat OpenCV (la mémoire WebAssembly n'est pas gérée automatiquement). */
function free(...mats) {
  for (const mat of mats) {
    if (mat && !mat.isDeleted()) mat.delete();
  }
}

/** Range 4 points : haut-gauche, haut-droit, bas-droit, bas-gauche. */
export function orderCorners(points) {
  const sums = points.map(([x, y]) => x + y);
  const diffs = points.map(([x, y]) => y - x);
  const argmin = (values) => values.indexOf(Math.min(...values));
  const argmax = (values) => values.indexOf(Math.max(...values));
  return [points[argmin(sums)], points[argmin(diffs)], points[argmax(sums)], points[argmax(diffs)]];
}

/** Trouve les 4 coins de la grille dans une image binaire (le plus grand carré). */
function findGridCorners(cv, binary) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const minArea = CONFIG.minGridAreaRatio * binary.rows * binary.cols;

  const candidates = [];
  for (let i = 0; i < contours.size(); i += 1) {
    const contour = contours.get(i);
    candidates.push({ contour, area: cv.contourArea(contour) });
  }
  candidates.sort((a, b) => b.area - a.area);

  let corners = null;
  for (const { contour, area } of candidates) {
    if (corners || area < minArea) break;
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, 0.02 * cv.arcLength(contour, true), true);
    if (approx.rows === 4 && cv.isContourConvex(approx)) {
      const rect = cv.boundingRect(approx);
      const error = Math.abs(rect.width - rect.height) / Math.max(rect.width, rect.height);
      if (error <= CONFIG.maxAspectError) {
        const p = approx.data32S;
        corners = orderCorners([
          [p[0], p[1]],
          [p[2], p[3]],
          [p[4], p[5]],
          [p[6], p[7]],
        ]);
      }
    }
    free(approx);
  }
  for (const { contour } of candidates) free(contour);
  free(contours, hierarchy);
  if (!corners) throw new GridNotFoundError();
  return corners;
}

/** Détecte la grille dans une image RGBA et renvoie sa version redressée (niveaux de gris). */
export function detectGrid(cv, rgba) {
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const binary = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
  cv.adaptiveThreshold(
    blurred,
    binary,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    11,
    2,
  );
  try {
    const corners = findGridCorners(cv, binary);
    const size = CONFIG.warpedSize;
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat());
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, size - 1, 0, size - 1, size - 1, 0, size - 1]);
    const matrix = cv.getPerspectiveTransform(src, dst);
    const warped = new cv.Mat();
    cv.warpPerspective(gray, warped, matrix, new cv.Size(size, size));
    const pixels = new Uint8Array(warped.data);
    free(src, dst, matrix, warped);
    return { corners, warped: pixels, size };
  } finally {
    free(gray, blurred, binary);
  }
}

/** Découpe la grille redressée en 81 cases (marges rognées), chacune en Uint8Array. */
export function splitCells(warped, size) {
  const step = size / CONFIG.gridSize;
  const margin = pyRound(step * CONFIG.cellMarginRatio);
  const cells = [];
  for (let r = 0; r < CONFIG.gridSize; r += 1) {
    const row = [];
    for (let c = 0; c < CONFIG.gridSize; c += 1) {
      const y0 = pyRound(r * step) + margin;
      const y1 = pyRound((r + 1) * step) - margin;
      const x0 = pyRound(c * step) + margin;
      const x1 = pyRound((c + 1) * step) - margin;
      const width = x1 - x0;
      const height = y1 - y0;
      const data = new Uint8Array(width * height);
      for (let y = 0; y < height; y += 1) {
        data.set(warped.subarray((y0 + y) * size + x0, (y0 + y) * size + x1), y * width);
      }
      row.push({ data, width, height });
    }
    cells.push(row);
  }
  return cells;
}

/** Masque « encre » d'une case, avec un seuil propre à la case (comme cells.ink_mask). */
export function inkMask(cell) {
  const { data } = cell;
  let darkest = 255;
  let lightest = 0;
  for (const v of data) {
    if (v < darkest) darkest = v;
    if (v > lightest) lightest = v;
  }
  const mask = new Uint8Array(data.length);
  if (data.length === 0 || lightest - darkest < CONFIG.minContrast) return mask;
  const level = (darkest + lightest) / 2;
  for (let i = 0; i < data.length; i += 1) mask[i] = data[i] < level ? 1 : 0;
  return mask;
}

export function isEmpty(cell) {
  const mask = inkMask(cell);
  let ink = 0;
  for (const v of mask) ink += v;
  return mask.length === 0 || ink / mask.length < CONFIG.emptyCellInkRatio;
}

/** Isole le chiffre d'une case et le normalise au format MNIST (28x28, blanc sur noir). */
export function extractDigit(cv, cell) {
  const { width, height } = cell;
  const mask = inkMask(cell);
  const binaryData = mask.map((v) => (v ? 255 : 0));
  const binary = cv.matFromArray(height, width, cv.CV_8UC1, binaryData);
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const count = cv.connectedComponentsWithStats(binary, labels, stats, centroids, 8, cv.CV_32S);

  const keepLabel = new Uint8Array(count);
  const s = stats.data32S;
  for (let label = 1; label < count; label += 1) {
    const [x, y, w, h, area] = s.subarray(label * 5, label * 5 + 5);
    const touchesBorder = x === 0 || y === 0 || x + w === width || y + h === height;
    const isLine = touchesBorder && (w > 0.8 * width || h > 0.8 * height);
    if (area >= 0.002 * width * height && !isLine) keepLabel[label] = 1;
  }

  const lab = labels.data32S;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  const kept = new Uint8Array(width * height);
  for (let i = 0; i < lab.length; i += 1) {
    if (keepLabel[lab[i]]) {
      kept[i] = 255;
      const x = i % width;
      const y = Math.floor(i / width);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  free(binary, labels, stats, centroids);
  if (maxX < 0) return null;

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;
  const crop = new Uint8Array(cropW * cropH);
  for (let y = 0; y < cropH; y += 1) {
    crop.set(kept.subarray((minY + y) * width + minX, (minY + y) * width + maxX + 1), y * cropW);
  }

  const size = CONFIG.digitSize;
  const box = pyRound((CONFIG.digitBox * size) / 28);
  const scale = box / Math.max(cropW, cropH);
  const newW = Math.max(1, pyRound(cropW * scale));
  const newH = Math.max(1, pyRound(cropH * scale));
  const cropMat = cv.matFromArray(cropH, cropW, cv.CV_8UC1, crop);
  const resized = new cv.Mat();
  cv.resize(cropMat, resized, new cv.Size(newW, newH), 0, 0, cv.INTER_AREA);

  const out = new Uint8Array(size * size);
  const top = Math.floor((size - newH) / 2);
  const left = Math.floor((size - newW) / 2);
  const r = resized.data;
  for (let y = 0; y < newH; y += 1) {
    out.set(r.subarray(y * newW, (y + 1) * newW), (top + y) * size + left);
  }
  free(cropMat, resized);
  return out;
}

/** Charge le CNN (fichier ONNX : URL, chemin ou octets). */
export async function loadModel(ort, source) {
  return ort.InferenceSession.create(source, { executionProviders: ["wasm"] });
}

/** Reconnaît une liste de chiffres normalisés avec le CNN. */
export async function classifyDigits(ort, session, digits) {
  if (!digits.length) return [];
  const pixels = CONFIG.digitSize * CONFIG.digitSize;
  const input = new Float32Array(digits.length * pixels);
  digits.forEach((digit, k) => {
    for (let i = 0; i < pixels; i += 1) input[k * pixels + i] = digit[i] / 255;
  });
  const tensor = new ort.Tensor("float32", input, [digits.length, 1, CONFIG.digitSize, CONFIG.digitSize]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  const logits = output[session.outputNames[0]].data;
  const classes = logits.length / digits.length;
  return digits.map((_, k) => {
    const row = Array.from(logits.subarray(k * classes, (k + 1) * classes));
    const max = Math.max(...row);
    const exp = row.map((v) => Math.exp(v - max));
    const total = exp.reduce((a, b) => a + b, 0);
    const best = row.indexOf(max);
    return { digit: best + 1, confidence: exp[best] / total };
  });
}

/**
 * Lit toute la grille d'une image RGBA.
 *
 * @returns {Promise<{grid: number[][], confidence: number[][], filled: boolean[][],
 *   corners: number[][], warped: Uint8Array, size: number, timings: Record<string, number>}>}
 */
export async function readGrid(cv, ort, session, rgba) {
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  const timings = {};
  let t = now();
  const { corners, warped, size } = detectGrid(cv, rgba);
  timings.grille = now() - t;

  t = now();
  const cells = splitCells(warped, size);
  const filled = cells.map((row) => row.map((cell) => !isEmpty(cell)));
  const positions = [];
  const digits = [];
  cells.forEach((row, r) =>
    row.forEach((cell, c) => {
      const digit = extractDigit(cv, cell);
      if (digit) {
        positions.push([r, c]);
        digits.push(digit);
      }
    }),
  );
  timings.cases = now() - t;

  t = now();
  const predictions = await classifyDigits(ort, session, digits);
  timings.chiffres = now() - t;

  const grid = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const confidence = Array.from({ length: 9 }, () => new Array(9).fill(1));
  positions.forEach(([r, c], k) => {
    grid[r][c] = predictions[k].digit;
    confidence[r][c] = predictions[k].confidence;
  });
  return { grid, confidence, filled, corners, warped, size, timings };
}
