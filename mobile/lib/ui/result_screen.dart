import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../core/pipeline.dart';
import '../core/solver.dart';
import 'sudoku_board.dart';
import 'theme.dart';

/// En dessous de cette confiance, un chiffre lu est signalé comme douteux.
const double doubtThreshold = 0.9;

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.reading, required this.photo});

  final GridReading reading;
  final Uint8List photo;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  late final Grid _puzzle = copyGrid(widget.reading.grid);
  late final List<List<bool>> _doubtful = [
    for (var r = 0; r < 9; r++)
      [
        for (var c = 0; c < 9; c++)
          widget.reading.grid[r][c] != 0 && widget.reading.confidence[r][c] < doubtThreshold,
      ],
  ];
  final _revealed = <(int, int)>{};
  final _random = Random();
  Grid? _solution;
  Set<(int, int)> _conflicts = {};
  bool _showAll = false;
  String _message = '';

  @override
  void initState() {
    super.initState();
    _solveCurrent(initial: true);
  }

  int get _givenCount => _puzzle.expand((row) => row).where((v) => v != 0).length;

  void _solveCurrent({bool initial = false}) {
    _solution = null;
    _conflicts = {};
    try {
      _solution = solve(_puzzle);
      final doubts = _doubtful.expand((row) => row).where((d) => d).length;
      _message = initial
          ? "J'ai lu $_givenCount chiffres. Comparez avec votre grille : "
              'touchez une case pour corriger un chiffre.'
          : 'Grille corrigée.';
      if (initial && doubts > 0) {
        _message += ' Vérifiez surtout les chiffres en orange.';
      }
    } on InvalidGridError catch (error) {
      _conflicts = error.cells.toSet();
      _message = 'Un même chiffre apparaît deux fois (en rouge) : il a sans doute été mal lu. '
          'Touchez-le pour le corriger.';
    } on UnsolvableGridError {
      _message = "Cette grille n'a pas de solution : un chiffre a sans doute été mal lu. "
          'Comparez avec votre grille et touchez une case pour la corriger.';
    }
  }

  List<List<int>> get _displayed => [
        for (var r = 0; r < 9; r++)
          [
            for (var c = 0; c < 9; c++)
              if (_puzzle[r][c] != 0)
                _puzzle[r][c]
              else if (_solution != null && (_showAll || _revealed.contains((r, c))))
                _solution![r][c]
              else
                0,
          ],
      ];

  List<List<CellKind>> get _kinds => [
        for (var r = 0; r < 9; r++)
          [
            for (var c = 0; c < 9; c++)
              if (_conflicts.contains((r, c)))
                CellKind.conflict
              else if (_puzzle[r][c] != 0 && _doubtful[r][c])
                CellKind.doubtful
              else if (_puzzle[r][c] != 0)
                CellKind.given
              else if (_solution != null && (_showAll || _revealed.contains((r, c))))
                CellKind.found
              else
                CellKind.empty,
          ],
      ];

  void _hint() {
    final solution = _solution;
    if (solution == null) return;
    final hidden = [
      for (var r = 0; r < 9; r++)
        for (var c = 0; c < 9; c++)
          if (_puzzle[r][c] == 0 && !_revealed.contains((r, c))) (r, c),
    ];
    if (hidden.isEmpty || _showAll) return;
    final (r, c) = hidden[_random.nextInt(hidden.length)];
    setState(() {
      _revealed.add((r, c));
      _message = 'Indice : un ${solution[r][c]} en ligne ${r + 1}, colonne ${c + 1}.';
    });
  }

  Future<void> _edit(int r, int c) async {
    if (_puzzle[r][c] == 0 && (_showAll || _revealed.contains((r, c)))) return;
    final choice = await showModalBottomSheet<int>(
      context: context,
      showDragHandle: true,
      builder: (context) => _NumberPad(row: r, col: c, current: _puzzle[r][c]),
    );
    if (choice == null || !mounted) return;
    setState(() {
      _puzzle[r][c] = choice;
      _doubtful[r][c] = false;
      _revealed.clear();
      _showAll = false;
      _solveCurrent();
    });
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final hasProblem = _solution == null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Votre grille', style: TextStyle(fontSize: 22)),
        backgroundColor: Palette.paper,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
          children: [
            SudokuBoard(values: _displayed, kinds: _kinds, onTap: _edit),
            const SizedBox(height: 16),
            Text(
              _message,
              style: text.bodyLarge?.copyWith(color: hasProblem ? Palette.error : Palette.ink),
            ),
            const SizedBox(height: 20),
            FilledButton.icon(
              onPressed: hasProblem || _showAll ? null : _hint,
              icon: const Icon(Icons.lightbulb_outline, size: 26),
              label: const Text('Un indice'),
            ),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: hasProblem ? null : () => setState(() => _showAll = !_showAll),
              child: Text(_showAll ? 'Cacher la solution' : 'Voir la solution'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.photo_camera_outlined),
              label: const Text('Nouvelle photo'),
            ),
            const SizedBox(height: 20),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              title: const Text('Voir ma photo', style: TextStyle(fontSize: 19)),
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.memory(widget.photo),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// Pavé numérique pour corriger une case.
class _NumberPad extends StatelessWidget {
  const _NumberPad({required this.row, required this.col, required this.current});

  final int row;
  final int col;
  final int current;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Ligne ${row + 1}, colonne ${col + 1}',
              style: Theme.of(context).textTheme.titleLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            GridView.count(
              crossAxisCount: 3,
              shrinkWrap: true,
              mainAxisSpacing: 10,
              crossAxisSpacing: 10,
              childAspectRatio: 1.6,
              physics: const NeverScrollableScrollPhysics(),
              children: [
                for (var d = 1; d <= 9; d++)
                  FilledButton.tonal(
                    onPressed: () => Navigator.of(context).pop(d),
                    style: FilledButton.styleFrom(
                      minimumSize: Size.zero,
                      backgroundColor: d == current ? Palette.cobalt : Palette.cobaltSoft,
                      foregroundColor: d == current ? Colors.white : Palette.cobalt,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    child: Text('$d', style: const TextStyle(fontSize: 30)),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            OutlinedButton(
              onPressed: () => Navigator.of(context).pop(0),
              child: const Text('Vider la case'),
            ),
          ],
        ),
      ),
    );
  }
}
