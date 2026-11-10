// Le pipeline Dart doit lire les captures du jeu de test comme la version Python
// (même vérité terrain : tests/data/grids.json à la racine du dépôt).

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sudoku_photo/core/cnn.dart';
import 'package:sudoku_photo/core/digits.dart';
import 'package:sudoku_photo/core/gray_image.dart';
import 'package:sudoku_photo/core/grid_detector.dart';
import 'package:sudoku_photo/core/pipeline.dart';
import 'package:sudoku_photo/core/solver.dart';

const dataDir = '../tests/data';

void main() {
  final cnn = DigitCnn.fromBytes(
    ByteData.sublistView(File('assets/digits_cnn.bin').readAsBytesSync()),
  );
  final annotations =
      (jsonDecode(File('$dataDir/grids.json').readAsStringSync()) as Map<String, dynamic>)
        ..removeWhere((name, _) => name.startsWith('_'));

  test('le jeu de test contient plusieurs sites', () {
    final sites = annotations.values.map((e) => (e as Map)['site']).toSet();
    expect(sites.length, greaterThanOrEqualTo(3));
  });

  annotations.forEach((name, entry) {
    test('lit $name comme la version Python', () {
      final expected = parseGrid(((entry as Map)['grid'] as List).join());
      final reading = readPhoto(File('$dataDir/$name').readAsBytesSync(), cnn);
      expect(reading.grid, expected);
      expect(solve(reading.grid), isNotNull);
      for (var r = 0; r < 9; r++) {
        for (var c = 0; c < 9; c++) {
          expect(reading.confidence[r][c], greaterThan(0.5));
        }
      }
    });
  });

  test('une image sans grille lève GridNotFoundError', () {
    final blank = GrayImage(300, 300, Uint8List(300 * 300)..fillRange(0, 300 * 300, 255));
    expect(() => readGrid(blank, cnn), throwsA(isA<GridNotFoundError>()));
  });

  test('une case vide ne donne aucun chiffre', () {
    final cell = GrayImage(38, 38, Uint8List(38 * 38)..fillRange(0, 38 * 38, 255));
    expect(extractDigit(cell), isNull);
    expect(isEmptyCell(cell), isTrue);
  });

  test('pyRound arrondit les ,5 vers le pair, comme Python', () {
    expect([0.5, 1.5, 2.5, 2.4, 2.6, 12.5].map(pyRound), [0, 2, 2, 2, 3, 12]);
  });

  test("l'homographie envoie chaque coin au bon endroit", () {
    final src = <Point>[(0.0, 0.0), (449.0, 0.0), (449.0, 449.0), (0.0, 449.0)];
    final dst = <Point>[(10.0, 20.0), (400.0, 35.0), (420.0, 410.0), (5.0, 380.0)];
    final h = homography(src, dst);
    for (var i = 0; i < 4; i++) {
      final (x, y) = applyHomography(h, src[i].$1, src[i].$2);
      expect(x, closeTo(dst[i].$1, 1e-6));
      expect(y, closeTo(dst[i].$2, 1e-6));
    }
  });

  test('la réduction par surface conserve la moyenne', () {
    final img = GrayImage(4, 2, Uint8List.fromList([0, 100, 200, 255, 0, 100, 200, 255]));
    final small = resizeArea(img, 2, 1);
    expect(small.data, [50, 228]);
  });
}
