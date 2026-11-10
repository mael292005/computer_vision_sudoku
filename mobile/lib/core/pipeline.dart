// Enchaînement complet : photo -> grille -> cases -> chiffres.

import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'cnn.dart';
import 'digits.dart';
import 'gray_image.dart';
import 'grid_detector.dart';

/// Côté maximal de l'image traitée : une photo de 12 Mpx est d'abord réduite.
const int maxSide = 1600;

class GridReading {
  GridReading(this.grid, this.confidence, this.corners);

  /// Grille 9x9 lue, 0 pour les cases vides.
  final List<List<int>> grid;

  /// Confiance de chaque lecture (1 pour les cases vides).
  final List<List<double>> confidence;

  /// Coins de la grille dans l'image traitée.
  final List<Point> corners;
}

/// Décode une photo (JPEG, PNG…), applique l'orientation EXIF et passe en gris.
GrayImage decodeToGray(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw const FormatException("format d'image non reconnu");
  final oriented = img.bakeOrientation(decoded);
  final rgb = oriented.convert(format: img.Format.uint8, numChannels: 3);
  return GrayImage.fromRgb(rgb.width, rgb.height, rgb.getBytes(order: img.ChannelOrder.rgb));
}

/// Réduit l'image si son plus grand côté dépasse [maxSide].
GrayImage limitSize(GrayImage gray) {
  final longest = gray.width > gray.height ? gray.width : gray.height;
  if (longest <= maxSide) return gray;
  final scale = maxSide / longest;
  return resizeArea(gray, pyRound(gray.width * scale), pyRound(gray.height * scale));
}

/// Lit la grille d'une image en niveaux de gris.
///
/// Lève [GridNotFoundError] si aucune grille n'est trouvée.
GridReading readGrid(GrayImage image, DigitCnn cnn) {
  final gray = limitSize(image);
  final binary = adaptiveThresholdInv(gaussianBlur(gray, 5, 1.1));
  final corners = findGridCorners(binary);
  final cells = splitCells(warpGrid(gray, corners));

  final grid = List.generate(9, (_) => List.filled(9, 0));
  final confidence = List.generate(9, (_) => List.filled(9, 1.0));
  for (var r = 0; r < 9; r++) {
    for (var c = 0; c < 9; c++) {
      final digit = extractDigit(cells[r][c]);
      if (digit == null) continue;
      final prediction = cnn.predict(digit);
      grid[r][c] = prediction.digit;
      confidence[r][c] = prediction.confidence;
    }
  }
  return GridReading(grid, confidence, corners);
}

/// Décode une photo puis lit la grille.
GridReading readPhoto(Uint8List bytes, DigitCnn cnn) => readGrid(decodeToGray(bytes), cnn);
