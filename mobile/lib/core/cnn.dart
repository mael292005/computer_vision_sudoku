// Le CNN de reconnaissance des chiffres, recodé en Dart pur.
//
// Même architecture que training/train_cnn.py :
//   conv 3x3 (1 -> 16) + ReLU + max-pooling 2x2   28x28 -> 14x14
//   conv 3x3 (16 -> 32) + ReLU + max-pooling 2x2  14x14 -> 7x7
//   dense 1568 -> 32 + ReLU, puis dense 32 -> 9 (chiffres 1 à 9)
// Les poids viennent de assets/digits_cnn.bin, exporté depuis le modèle ONNX
// par training/export_mobile_weights.py.

import 'dart:math' as math;
import 'dart:typed_data';

class Prediction {
  const Prediction(this.digit, this.confidence);

  final int digit;
  final double confidence;
}

class DigitCnn {
  DigitCnn._(this._w);

  /// Charge les poids (float32 little-endian, dans l'ordre de l'export Python).
  factory DigitCnn.fromBytes(ByteData bytes) {
    const counts = [144, 16, 4608, 32, 50176, 32, 288, 9];
    final total = counts.reduce((a, b) => a + b);
    if (bytes.lengthInBytes != total * 4) {
      throw ArgumentError('fichier de poids invalide : ${bytes.lengthInBytes} octets');
    }
    final layers = <Float64List>[];
    var offset = 0;
    for (final count in counts) {
      final layer = Float64List(count);
      for (var i = 0; i < count; i++) {
        layer[i] = bytes.getFloat32(offset, Endian.little);
        offset += 4;
      }
      layers.add(layer);
    }
    return DigitCnn._(layers);
  }

  final List<Float64List> _w;

  /// Convolution 3x3 avec marge de 1, suivie de ReLU.
  static Float64List _convRelu(
      Float64List input, int cIn, int size, Float64List weight, Float64List bias, int cOut) {
    final out = Float64List(cOut * size * size);
    for (var o = 0; o < cOut; o++) {
      for (var y = 0; y < size; y++) {
        for (var x = 0; x < size; x++) {
          var acc = bias[o];
          for (var i = 0; i < cIn; i++) {
            for (var ky = 0; ky < 3; ky++) {
              final iy = y + ky - 1;
              if (iy < 0 || iy >= size) continue;
              for (var kx = 0; kx < 3; kx++) {
                final ix = x + kx - 1;
                if (ix < 0 || ix >= size) continue;
                acc += weight[((o * cIn + i) * 3 + ky) * 3 + kx] * input[(i * size + iy) * size + ix];
              }
            }
          }
          out[(o * size + y) * size + x] = acc > 0 ? acc : 0;
        }
      }
    }
    return out;
  }

  static Float64List _maxPool(Float64List input, int channels, int size) {
    final half = size ~/ 2;
    final out = Float64List(channels * half * half);
    for (var c = 0; c < channels; c++) {
      for (var y = 0; y < half; y++) {
        for (var x = 0; x < half; x++) {
          var best = double.negativeInfinity;
          for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
              best = math.max(best, input[(c * size + 2 * y + dy) * size + 2 * x + dx]);
            }
          }
          out[(c * half + y) * half + x] = best;
        }
      }
    }
    return out;
  }

  static Float64List _dense(Float64List input, Float64List weight, Float64List bias, {bool relu = false}) {
    final out = Float64List(bias.length);
    for (var o = 0; o < bias.length; o++) {
      var acc = bias[o];
      final row = o * input.length;
      for (var i = 0; i < input.length; i++) {
        acc += weight[row + i] * input[i];
      }
      out[o] = relu && acc < 0 ? 0 : acc;
    }
    return out;
  }

  /// Sorties brutes (logits) pour un chiffre 28x28.
  Float64List logits(Uint8List digit) {
    var x = Float64List(28 * 28);
    for (var i = 0; i < x.length; i++) {
      x[i] = digit[i] / 255;
    }
    x = _maxPool(_convRelu(x, 1, 28, _w[0], _w[1], 16), 16, 28);
    x = _maxPool(_convRelu(x, 16, 14, _w[2], _w[3], 32), 32, 14);
    x = _dense(x, _w[4], _w[5], relu: true);
    return _dense(x, _w[6], _w[7]);
  }

  Prediction predict(Uint8List digit) {
    final out = logits(digit);
    var best = 0;
    for (var i = 1; i < out.length; i++) {
      if (out[i] > out[best]) best = i;
    }
    var total = 0.0;
    for (final v in out) {
      total += math.exp(v - out[best]);
    }
    return Prediction(best + 1, 1 / total);
  }
}
