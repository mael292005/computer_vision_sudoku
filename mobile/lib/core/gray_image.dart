// Image en niveaux de gris et opérations de base, en Dart pur (sans OpenCV).
//
// Ces fonctions reproduisent celles d'OpenCV utilisées par la version Python :
// conversion en gris, réduction par moyenne de surface (INTER_AREA), flou
// gaussien et seuillage adaptatif gaussien.

import 'dart:math' as math;
import 'dart:typed_data';

/// Arrondi comme Python 3 : les ,5 vont vers l'entier pair.
int pyRound(double x) {
  final floor = x.floorToDouble();
  final diff = x - floor;
  if (diff > 0.5) return floor.toInt() + 1;
  if (diff < 0.5) return floor.toInt();
  return floor.toInt().isEven ? floor.toInt() : floor.toInt() + 1;
}

int _clampByte(double v) {
  final r = (v + 0.5).floor();
  return r < 0 ? 0 : (r > 255 ? 255 : r);
}

class GrayImage {
  GrayImage(this.width, this.height, [Uint8List? data])
      : data = data ?? Uint8List(width * height) {
    if (this.data.length != width * height) {
      throw ArgumentError('taille incohérente : ${this.data.length} != $width x $height');
    }
  }

  /// Conversion depuis des octets RGB (3 par pixel), comme cv2.COLOR_RGB2GRAY.
  factory GrayImage.fromRgb(int width, int height, Uint8List rgb) {
    final out = Uint8List(width * height);
    for (var i = 0, j = 0; i < out.length; i++, j += 3) {
      out[i] = _clampByte(0.299 * rgb[j] + 0.587 * rgb[j + 1] + 0.114 * rgb[j + 2]);
    }
    return GrayImage(width, height, out);
  }

  final int width;
  final int height;
  final Uint8List data;

  int at(int x, int y) => data[y * width + x];

  /// Sous-image (copie) de la zone [x0, x1[ x [y0, y1[.
  GrayImage crop(int x0, int y0, int x1, int y1) {
    final w = x1 - x0;
    final h = y1 - y0;
    final out = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      out.setRange(y * w, (y + 1) * w, data, (y0 + y) * width + x0);
    }
    return GrayImage(w, h, out);
  }

  /// Plus petite et plus grande valeur des pixels.
  (int, int) minMax() {
    var lo = 255;
    var hi = 0;
    for (final v in data) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return (lo, hi);
  }
}

/// Coefficients de réduction par moyenne de surface, pour un axe.
List<List<(int, double)>> _areaWeights(int nOut, int nIn) {
  final scale = nIn / nOut;
  final result = <List<(int, double)>>[];
  for (var i = 0; i < nOut; i++) {
    final a = i * scale;
    final b = (i + 1) * scale;
    final j0 = a.floor();
    final j1 = math.min(b.ceil(), nIn);
    final weights = <(int, double)>[];
    var total = 0.0;
    for (var j = j0; j < j1; j++) {
      final w = math.max(0.0, math.min(b, j + 1.0) - math.max(a, j.toDouble()));
      if (w > 0) {
        weights.add((j, w));
        total += w;
      }
    }
    result.add([for (final (j, w) in weights) (j, w / total)]);
  }
  return result;
}

/// Échantillonnage bilinéaire en (x, y), coordonnées bornées à l'image.
double bilinear(GrayImage img, double x, double y) {
  final cx = x < 0 ? 0.0 : (x > img.width - 1 ? (img.width - 1).toDouble() : x);
  final cy = y < 0 ? 0.0 : (y > img.height - 1 ? (img.height - 1).toDouble() : y);
  final x0 = cx.floor();
  final y0 = cy.floor();
  final x1 = math.min(x0 + 1, img.width - 1);
  final y1 = math.min(y0 + 1, img.height - 1);
  final fx = cx - x0;
  final fy = cy - y0;
  final top = img.at(x0, y0) * (1 - fx) + img.at(x1, y0) * fx;
  final bottom = img.at(x0, y1) * (1 - fx) + img.at(x1, y1) * fx;
  return top * (1 - fy) + bottom * fy;
}

/// Redimensionne : moyenne de surface pour réduire, bilinéaire pour agrandir.
GrayImage resizeArea(GrayImage img, int newWidth, int newHeight) {
  final out = Uint8List(newWidth * newHeight);
  if (newWidth <= img.width && newHeight <= img.height) {
    final wy = _areaWeights(newHeight, img.height);
    final wx = _areaWeights(newWidth, img.width);
    // Passe horizontale puis verticale, en flottants.
    final tmp = Float64List(img.height * newWidth);
    for (var y = 0; y < img.height; y++) {
      final row = y * img.width;
      for (var x = 0; x < newWidth; x++) {
        var acc = 0.0;
        for (final (j, w) in wx[x]) {
          acc += img.data[row + j] * w;
        }
        tmp[y * newWidth + x] = acc;
      }
    }
    for (var y = 0; y < newHeight; y++) {
      for (var x = 0; x < newWidth; x++) {
        var acc = 0.0;
        for (final (j, w) in wy[y]) {
          acc += tmp[j * newWidth + x] * w;
        }
        out[y * newWidth + x] = _clampByte(acc);
      }
    }
  } else {
    for (var y = 0; y < newHeight; y++) {
      final sy = (y + 0.5) * img.height / newHeight - 0.5;
      for (var x = 0; x < newWidth; x++) {
        final sx = (x + 0.5) * img.width / newWidth - 0.5;
        out[y * newWidth + x] = _clampByte(bilinear(img, sx, sy));
      }
    }
  }
  return GrayImage(newWidth, newHeight, out);
}

List<double> _gaussianKernel(int size, double sigma) {
  final half = (size - 1) / 2;
  final k = [for (var i = 0; i < size; i++) math.exp(-((i - half) * (i - half)) / (2 * sigma * sigma))];
  final total = k.reduce((a, b) => a + b);
  return [for (final v in k) v / total];
}

/// Indice de bord : `reflect` = miroir sans répéter le bord (BORDER_REFLECT_101),
/// sinon on répète le pixel du bord (BORDER_REPLICATE).
int _border(int i, int n, {required bool reflect}) {
  if (i >= 0 && i < n) return i;
  if (!reflect || n == 1) return i < 0 ? 0 : n - 1;
  var j = i;
  while (j < 0 || j >= n) {
    j = j < 0 ? -j : 2 * (n - 1) - j;
  }
  return j;
}

/// Convolution gaussienne séparable, résultat en flottants.
Float64List _separable(GrayImage img, List<double> kernel, {required bool reflect}) {
  final r = kernel.length ~/ 2;
  final w = img.width;
  final h = img.height;
  final tmp = Float64List(w * h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var acc = 0.0;
      for (var k = 0; k < kernel.length; k++) {
        acc += kernel[k] * img.data[y * w + _border(x + k - r, w, reflect: reflect)];
      }
      tmp[y * w + x] = acc;
    }
  }
  final out = Float64List(w * h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      var acc = 0.0;
      for (var k = 0; k < kernel.length; k++) {
        acc += kernel[k] * tmp[_border(y + k - r, h, reflect: reflect) * w + x];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

/// Flou gaussien (cv2.GaussianBlur, bords en miroir).
GrayImage gaussianBlur(GrayImage img, int size, double sigma) {
  final values = _separable(img, _gaussianKernel(size, sigma), reflect: true);
  final out = Uint8List(values.length);
  for (var i = 0; i < values.length; i++) {
    out[i] = _clampByte(values[i]);
  }
  return GrayImage(img.width, img.height, out);
}

/// Seuillage adaptatif gaussien inversé : l'encre devient blanche (255).
///
/// Comme cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, block, c).
GrayImage adaptiveThresholdInv(GrayImage img, {int block = 11, int c = 2}) {
  final sigma = 0.3 * ((block - 1) * 0.5 - 1) + 0.8;
  final mean = _separable(img, _gaussianKernel(block, sigma), reflect: false);
  final out = Uint8List(img.data.length);
  for (var i = 0; i < out.length; i++) {
    final m = (mean[i] + 0.5).floorToDouble();
    out[i] = img.data[i] - m > -c ? 0 : 255;
  }
  return GrayImage(img.width, img.height, out);
}
