// Découpe des cases et extraction des chiffres : portage de cells.py et
// de digits.extract_digit, mêmes réglages que src/sudoku_vision/config.py.

import 'dart:math' as math;
import 'dart:typed_data';

import 'grid_detector.dart';
import 'gray_image.dart';

const double cellMarginRatio = 0.12;
const double emptyCellInkRatio = 0.01;
const int minContrast = 60;
const int digitSize = 28;
const int digitBox = 20;

/// Découpe la grille redressée en 9x9 cases, marges rognées.
List<List<GrayImage>> splitCells(GrayImage warped) {
  final step = warped.width / 9;
  final margin = pyRound(step * cellMarginRatio);
  return [
    for (var r = 0; r < 9; r++)
      [
        for (var c = 0; c < 9; c++)
          warped.crop(
            pyRound(c * step) + margin,
            pyRound(r * step) + margin,
            pyRound((c + 1) * step) - margin,
            pyRound((r + 1) * step) - margin,
          ),
      ],
  ];
}

/// Masque « encre » (1 = encre), avec un seuil propre à la case.
Uint8List inkMask(GrayImage cell) {
  final mask = Uint8List(cell.data.length);
  if (cell.data.isEmpty) return mask;
  final (lo, hi) = cell.minMax();
  if (hi - lo < minContrast) return mask;
  final level = (lo + hi) / 2;
  for (var i = 0; i < mask.length; i++) {
    mask[i] = cell.data[i] < level ? 1 : 0;
  }
  return mask;
}

bool isEmptyCell(GrayImage cell) {
  final mask = inkMask(cell);
  if (mask.isEmpty) return true;
  var ink = 0;
  for (final v in mask) {
    ink += v;
  }
  return ink / mask.length < emptyCellInkRatio;
}

/// Isole le chiffre d'une case, au format MNIST (28x28, blanc sur noir).
///
/// Renvoie `null` si la case ne contient pas de chiffre.
Uint8List? extractDigit(GrayImage cell) {
  final w = cell.width;
  final h = cell.height;
  final mask = inkMask(cell);
  final (labels, components) = labelComponents(mask, w, h);

  final keep = List<bool>.filled(components.length + 1, false);
  for (var i = 0; i < components.length; i++) {
    final c = components[i];
    final touchesBorder = c.minX == 0 || c.minY == 0 || c.maxX == w - 1 || c.maxY == h - 1;
    final isLine = touchesBorder && (c.width > 0.8 * w || c.height > 0.8 * h);
    if (c.count >= 0.002 * w * h && !isLine) keep[i + 1] = true;
  }

  var minX = w, minY = h, maxX = -1, maxY = -1;
  final kept = Uint8List(w * h);
  for (var i = 0; i < labels.length; i++) {
    if (!keep[labels[i]]) continue;
    kept[i] = 255;
    final x = i % w;
    final y = i ~/ w;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (maxX < 0) return null;

  final digit = GrayImage(w, h, kept).crop(minX, minY, maxX + 1, maxY + 1);
  final box = pyRound(digitBox * digitSize / 28);
  final scale = box / (digit.width > digit.height ? digit.width : digit.height);
  final newW = math.max(1, math.min(digitSize, pyRound(digit.width * scale)));
  final newH = math.max(1, math.min(digitSize, pyRound(digit.height * scale)));
  final resized = resizeArea(digit, newW, newH);

  final out = Uint8List(digitSize * digitSize);
  final top = (digitSize - newH) ~/ 2;
  final left = (digitSize - newW) ~/ 2;
  for (var y = 0; y < newH; y++) {
    out.setRange((top + y) * digitSize + left, (top + y) * digitSize + left + newW,
        resized.data, y * newW);
  }
  return out;
}
