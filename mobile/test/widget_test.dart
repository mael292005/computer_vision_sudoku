import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sudoku_photo/core/pipeline.dart';
import 'package:sudoku_photo/core/solver.dart';
import 'package:sudoku_photo/main.dart';
import 'package:sudoku_photo/ui/result_screen.dart';
import 'package:sudoku_photo/ui/theme.dart';

const puzzle = '530070000600195000098000060800060003400803001700020006060000280000419005000080079';

GridReading reading(Grid grid, {double confidence = 1}) => GridReading(
      grid,
      List.generate(9, (_) => List.filled(9, confidence)),
      const [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    );

Future<void> showResult(WidgetTester tester, GridReading reading) async {
  tester.view.physicalSize = const Size(1080, 2400);
  tester.view.devicePixelRatio = 2.5;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: buildTheme(),
    home: ResultScreen(reading: reading, photo: Uint8List(0)),
  ));
}

void main() {
  testWidgets("l'accueil propose de prendre une photo", (tester) async {
    await tester.pumpWidget(const SudokuPhotoApp());
    expect(find.text('Sudoku Photo'), findsOneWidget);
    expect(find.text('Prendre la grille en photo'), findsOneWidget);
    expect(find.text('Choisir une photo'), findsOneWidget);
  });

  testWidgets('la grille lue est affichée et un indice révèle une case', (tester) async {
    final grid = parseGrid(puzzle);
    await showResult(tester, reading(grid));
    final given = grid.expand((row) => row).where((v) => v != 0).length;
    expect(find.textContaining("J'ai lu $given chiffres"), findsOneWidget);

    await tester.tap(find.text('Un indice'));
    await tester.pump();
    expect(find.textContaining('Indice : un '), findsOneWidget);
  });

  testWidgets('les chiffres en conflit sont signalés', (tester) async {
    final grid = parseGrid(puzzle);
    grid[0][2] = 5; // deux 5 sur la première ligne
    await showResult(tester, reading(grid));
    expect(find.textContaining('apparaît deux fois'), findsOneWidget);
  });

  testWidgets('les chiffres douteux sont signalés', (tester) async {
    await showResult(tester, reading(parseGrid(puzzle), confidence: 0.5));
    expect(find.textContaining('chiffres en orange'), findsOneWidget);
  });
}
