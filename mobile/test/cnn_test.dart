// Le CNN recodé en Dart doit donner les mêmes sorties que le modèle ONNX.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sudoku_photo/core/cnn.dart';

void main() {
  final cnn = DigitCnn.fromBytes(
    ByteData.sublistView(File('assets/digits_cnn.bin').readAsBytesSync()),
  );
  final cases = (jsonDecode(File('test/fixtures/cnn_reference.json').readAsStringSync()) as List)
      .cast<Map<String, dynamic>>();

  test('il y a des chiffres de référence', () {
    expect(cases.length, greaterThanOrEqualTo(20));
  });

  for (final reference in cases) {
    test('mêmes sorties que le modèle ONNX : ${reference['source']}', () {
      final pixels = Uint8List.fromList((reference['pixels'] as List).cast<int>());
      final expected = (reference['logits'] as List).cast<num>();
      final logits = cnn.logits(pixels);
      for (var i = 0; i < expected.length; i++) {
        expect(logits[i], closeTo(expected[i].toDouble(), 1e-3));
      }
      final prediction = cnn.predict(pixels);
      expect(prediction.digit, reference['expected_digit']);
      expect(prediction.confidence, inInclusiveRange(0.0, 1.0));
    });
  }

  test('rejette un fichier de poids de mauvaise taille', () {
    expect(() => DigitCnn.fromBytes(ByteData(12)), throwsArgumentError);
  });
}
