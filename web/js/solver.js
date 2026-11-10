// Solveur de Sudoku : portage direct de src/sudoku_vision/solver.py.
//
// Backtracking guidé par l'heuristique MRV : on remplit d'abord la case qui a
// le moins de candidats. Les candidats de chaque ligne, colonne et bloc sont
// stockés en masques de bits (bit d = chiffre d déjà utilisé).
//
// Une grille est un tableau de 9 tableaux de 9 entiers, 0 = case vide.

const SIZE = 9;
const BOX = 3;
const ALL_DIGITS = 0b1111111110; // bits 1 à 9

export class InvalidGridError extends Error {
  /** @param {string} message @param {Array<[number, number]>} [cells] cases en conflit */
  constructor(message, cells = []) {
    super(message);
    this.name = "InvalidGridError";
    this.cells = cells;
  }
}

export class UnsolvableGridError extends Error {
  constructor(message) {
    super(message);
    this.name = "UnsolvableGridError";
  }
}

const boxIndex = (r, c) => Math.floor(r / BOX) * BOX + Math.floor(c / BOX);

function popcount(mask) {
  let count = 0;
  while (mask) {
    mask &= mask - 1;
    count += 1;
  }
  return count;
}

/** Renvoie la liste des cases en conflit (même chiffre deux fois dans une ligne, colonne ou bloc). */
export function findConflicts(grid) {
  const conflicts = new Set();
  const groups = [];
  for (let i = 0; i < SIZE; i += 1) {
    const row = [];
    const col = [];
    const box = [];
    for (let j = 0; j < SIZE; j += 1) {
      row.push([i, j]);
      col.push([j, i]);
      box.push([Math.floor(i / BOX) * BOX + Math.floor(j / BOX), (i % BOX) * BOX + (j % BOX)]);
    }
    groups.push(row, col, box);
  }
  for (const group of groups) {
    const seen = new Map();
    for (const [r, c] of group) {
      const value = grid[r][c];
      if (!value) continue;
      if (seen.has(value)) {
        conflicts.add(`${r},${c}`);
        conflicts.add(seen.get(value));
      } else {
        seen.set(value, `${r},${c}`);
      }
    }
  }
  return [...conflicts].map((key) => key.split(",").map(Number));
}

/** Vérifie la forme de la grille et l'absence de doublons. */
export function validateGrid(grid) {
  if (!Array.isArray(grid) || grid.length !== SIZE || grid.some((row) => row.length !== SIZE)) {
    throw new InvalidGridError("la grille doit faire 9x9");
  }
  for (let r = 0; r < SIZE; r += 1) {
    for (let c = 0; c < SIZE; c += 1) {
      const value = grid[r][c];
      if (!Number.isInteger(value) || value < 0 || value > SIZE) {
        throw new InvalidGridError(`valeur invalide ${value} en (${r}, ${c})`);
      }
    }
  }
  const conflicts = findConflicts(grid);
  if (conflicts.length) {
    throw new InvalidGridError("un même chiffre apparaît deux fois", conflicts);
  }
}

/** Résout une grille ; la grille d'entrée n'est pas modifiée. */
export function solve(grid) {
  validateGrid(grid);
  const board = grid.map((row) => [...row]);
  const rows = new Array(SIZE).fill(0);
  const cols = new Array(SIZE).fill(0);
  const boxes = new Array(SIZE).fill(0);
  const empties = [];

  for (let r = 0; r < SIZE; r += 1) {
    for (let c = 0; c < SIZE; c += 1) {
      const value = board[r][c];
      if (value) {
        const bit = 1 << value;
        rows[r] |= bit;
        cols[c] |= bit;
        boxes[boxIndex(r, c)] |= bit;
      } else {
        empties.push([r, c]);
      }
    }
  }

  const candidates = (r, c) => ALL_DIGITS & ~(rows[r] | cols[c] | boxes[boxIndex(r, c)]);

  function backtrack() {
    if (!empties.length) return true;

    // MRV : la case vide qui a le moins de candidats.
    let bestIndex = -1;
    let bestMask = 0;
    let bestCount = SIZE + 1;
    for (let i = 0; i < empties.length; i += 1) {
      const [r, c] = empties[i];
      const mask = candidates(r, c);
      const count = popcount(mask);
      if (count < bestCount) {
        bestIndex = i;
        bestMask = mask;
        bestCount = count;
        if (count <= 1) break;
      }
    }
    if (bestCount === 0) return false;

    const [r, c] = empties.splice(bestIndex, 1)[0];
    const b = boxIndex(r, c);
    let mask = bestMask;
    while (mask) {
      const bit = mask & -mask;
      mask ^= bit;
      board[r][c] = 31 - Math.clz32(bit);
      rows[r] |= bit;
      cols[c] |= bit;
      boxes[b] |= bit;
      if (backtrack()) return true;
      rows[r] ^= bit;
      cols[c] ^= bit;
      boxes[b] ^= bit;
    }
    board[r][c] = 0;
    empties.splice(bestIndex, 0, [r, c]);
    return false;
  }

  if (!backtrack()) throw new UnsolvableGridError("cette grille n'a pas de solution");
  return board;
}

/** Grille depuis 81 caractères (chiffres, 0 ou . pour les cases vides). */
export function parseGrid(text) {
  const cells = [...text].filter((ch) => !" \t\r\n|-+".includes(ch));
  if (cells.length !== SIZE * SIZE) {
    throw new InvalidGridError(`81 cases attendues, ${cells.length} trouvées`);
  }
  const values = cells.map((ch) => {
    if (ch === ".") return 0;
    if (/^[0-9]$/.test(ch)) return Number(ch);
    throw new InvalidGridError(`caractère inattendu « ${ch} »`);
  });
  return Array.from({ length: SIZE }, (_, r) => values.slice(r * SIZE, (r + 1) * SIZE));
}

export const emptyGrid = () => Array.from({ length: SIZE }, () => new Array(SIZE).fill(0));
