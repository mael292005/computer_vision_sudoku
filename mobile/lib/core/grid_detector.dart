// Détection de la grille dans une photo, puis redressement.
//
// 1. On cherche les composantes connexes de l'image seuillée : une grille de
//    Sudoku forme un seul bloc (cadre + lignes internes reliés).
// 2. Pour chaque grande composante, on prend ses 4 points extrêmes (x + y et
//    y - x minimum et maximum) : ce sont les coins, même si la photo est
//    inclinée ou prise de biais.
// 3. On garde la plus grande qui a des côtés de longueurs proches ET des lignes
//    internes à 1/3 et 2/3 : le bord d'une feuille ou d'un écran n'en a pas.
// 4. Une homographie « déplie » la grille en un carré vu de face.

import 'dart:math' as math;
import 'dart:typed_data';

import 'gray_image.dart';

const int warpedSize = 450;

class GridNotFoundError implements Exception {
  @override
  String toString() => "GridNotFoundError: aucune grille de Sudoku trouvée dans l'image";
}

/// Un point de l'image (coordonnées en pixels).
typedef Point = (double, double);

/// Statistiques d'une composante connexe.
class Component {
  Component(int x, int y)
      : minX = x,
        maxX = x,
        minY = y,
        maxY = y,
        _corners = List.filled(4, (x, y));

  int minX;
  int maxX;
  int minY;
  int maxY;
  int count = 0;

  // Points extrêmes : min(x+y), min(y-x), max(x+y), max(y-x).
  final List<(int, int)> _corners;

  int get width => maxX - minX + 1;
  int get height => maxY - minY + 1;

  void _add(int x, int y) {
    count++;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    // À égalité, le pixel le plus haut gagne (premier dans l'ordre de lecture).
    bool better(int value, int best, int yNew, int yBest, {required bool lower}) =>
        (lower ? value < best : value > best) || (value == best && yNew < yBest);
    final s = x + y;
    final d = y - x;
    final (tlx, tly) = _corners[0];
    if (better(s, tlx + tly, y, tly, lower: true)) _corners[0] = (x, y);
    final (trx, try_) = _corners[1];
    if (better(d, try_ - trx, y, try_, lower: true)) _corners[1] = (x, y);
    final (brx, bry) = _corners[2];
    if (better(s, brx + bry, y, bry, lower: false)) _corners[2] = (x, y);
    final (blx, bly) = _corners[3];
    if (better(d, bly - blx, y, bly, lower: false)) _corners[3] = (x, y);
  }

  /// Coins : haut-gauche, haut-droit, bas-droit, bas-gauche.
  List<Point> get quad => [for (final (x, y) in _corners) (x.toDouble(), y.toDouble())];
}

/// Étiquette les composantes 8-connexes des pixels non nuls de [mask].
///
/// Renvoie les étiquettes (0 = fond, k = composante k-1) et les composantes.
(Int32List, List<Component>) labelComponents(Uint8List mask, int width, int height) {
  final labels = Int32List(width * height);
  final components = <Component>[];
  final queue = Int32List(width * height);
  for (var start = 0; start < mask.length; start++) {
    if (mask[start] == 0 || labels[start] != 0) continue;
    final label = components.length + 1;
    final component = Component(start % width, start ~/ width);
    components.add(component);
    labels[start] = label;
    var head = 0;
    var tail = 0;
    queue[tail++] = start;
    while (head < tail) {
      final index = queue[head++];
      final x = index % width;
      final y = index ~/ width;
      component._add(x, y);
      for (var dy = -1; dy <= 1; dy++) {
        final ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (var dx = -1; dx <= 1; dx++) {
          final nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          final n = ny * width + nx;
          if (mask[n] != 0 && labels[n] == 0) {
            labels[n] = label;
            queue[tail++] = n;
          }
        }
      }
    }
  }
  return (labels, components);
}

/// Homographie 3x3 qui envoie les 4 points [src] sur les 4 points [dst].
List<double> homography(List<Point> src, List<Point> dst) {
  final a = List.generate(8, (_) => List<double>.filled(9, 0));
  for (var i = 0; i < 4; i++) {
    final (x, y) = src[i];
    final (u, v) = dst[i];
    a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, u];
    a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v];
  }
  // Élimination de Gauss avec pivot partiel.
  for (var col = 0; col < 8; col++) {
    var pivot = col;
    for (var r = col + 1; r < 8; r++) {
      if (a[r][col].abs() > a[pivot][col].abs()) pivot = r;
    }
    final tmp = a[col];
    a[col] = a[pivot];
    a[pivot] = tmp;
    if (a[col][col].abs() < 1e-12) throw GridNotFoundError();
    for (var r = 0; r < 8; r++) {
      if (r == col) continue;
      final f = a[r][col] / a[col][col];
      for (var k = col; k < 9; k++) {
        a[r][k] -= f * a[col][k];
      }
    }
  }
  return [for (var i = 0; i < 8; i++) a[i][8] / a[i][i], 1.0];
}

Point applyHomography(List<double> h, double x, double y) {
  final den = h[6] * x + h[7] * y + h[8];
  return ((h[0] * x + h[1] * y + h[2]) / den, (h[3] * x + h[4] * y + h[5]) / den);
}

List<Point> _unitSquare(int size) {
  final far = (size - 1).toDouble();
  return [(0.0, 0.0), (far, 0.0), (far, far), (0.0, far)];
}

double _quadArea(List<Point> q) {
  var sum = 0.0;
  for (var i = 0; i < 4; i++) {
    final (x1, y1) = q[i];
    final (x2, y2) = q[(i + 1) % 4];
    sum += x1 * y2 - y1 * x2;
  }
  return sum.abs() / 2;
}

double _sideRatio(List<Point> q) {
  final sides = [
    for (var i = 0; i < 4; i++)
      math.sqrt(math.pow(q[i].$1 - q[(i + 1) % 4].$1, 2) + math.pow(q[i].$2 - q[(i + 1) % 4].$2, 2)),
  ];
  return sides.reduce(math.max) / math.max(sides.reduce(math.min), 1);
}

/// Une grille a des lignes à 1/3 et 2/3 de sa largeur : on les cherche.
bool hasInnerLines(GrayImage binary, List<Point> quad, {int band = 6, double minInk = 0.6}) {
  const size = warpedSize;
  final h = homography(_unitSquare(size), quad);
  for (final k in [1, 2]) {
    final pos = k * (size - 1) / 3;
    for (final vertical in [true, false]) {
      var best = 0.0;
      for (var off = -band; off <= band; off++) {
        var ink = 0;
        var total = 0;
        for (var t = 8; t < size - 8; t++) {
          final (x, y) = vertical ? (pos + off, t.toDouble()) : (t.toDouble(), pos + off);
          final (sx, sy) = applyHomography(h, x, y);
          final ix = pyRound(sx);
          final iy = pyRound(sy);
          total++;
          if (ix >= 0 && iy >= 0 && ix < binary.width && iy < binary.height && binary.at(ix, iy) > 0) {
            ink++;
          }
        }
        best = math.max(best, ink / total);
      }
      if (best < minInk) return false;
    }
  }
  return true;
}

/// Trouve les 4 coins de la grille dans une image seuillée (encre = 255).
List<Point> findGridCorners(GrayImage binary) {
  final (_, components) = labelComponents(binary.data, binary.width, binary.height);
  final minArea = 0.01 * binary.width * binary.height;
  final candidates = <(double, List<Point>)>[];
  for (final c in components) {
    if (c.width * c.height < minArea) continue;
    final quad = c.quad;
    if (_sideRatio(quad) > 1.35) continue;
    candidates.add((_quadArea(quad), quad));
  }
  candidates.sort((a, b) => b.$1.compareTo(a.$1));
  for (final (_, quad) in candidates) {
    if (hasInnerLines(binary, quad)) return quad;
  }
  throw GridNotFoundError();
}

/// Redresse la zone [corners] de [gray] en un carré de [size] pixels.
GrayImage warpGrid(GrayImage gray, List<Point> corners, {int size = warpedSize}) {
  final h = homography(_unitSquare(size), corners);
  final out = Uint8List(size * size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final (sx, sy) = applyHomography(h, x.toDouble(), y.toDouble());
      final v = (bilinear(gray, sx, sy) + 0.5).floor();
      out[y * size + x] = v < 0 ? 0 : (v > 255 ? 255 : v);
    }
  }
  return GrayImage(size, size, out);
}
