import 'package:flutter/material.dart';

import 'theme.dart';

/// État d'affichage d'une case.
enum CellKind { given, doubtful, conflict, found, empty }

/// La grille 9x9 : chiffres lus en noir (police à empattements, comme le papier),
/// chiffres trouvés en bleu, chiffres douteux en orange, conflits en rouge.
class SudokuBoard extends StatelessWidget {
  const SudokuBoard({
    super.key,
    required this.values,
    required this.kinds,
    required this.onTap,
  });

  final List<List<int>> values;
  final List<List<CellKind>> kinds;
  final void Function(int row, int col) onTap;

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          border: Border.all(color: Palette.box, width: 2.5),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Column(
          children: [
            for (var r = 0; r < 9; r++)
              Expanded(
                child: Row(
                  children: [
                    for (var c = 0; c < 9; c++) Expanded(child: _cell(r, c)),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _cell(int r, int c) {
    final kind = kinds[r][c];
    final value = values[r][c];
    final background = switch (kind) {
      CellKind.conflict => Palette.errorSoft,
      CellKind.doubtful => Palette.doubtSoft,
      _ => Colors.transparent,
    };
    final style = switch (kind) {
      CellKind.found => const TextStyle(color: Palette.cobalt, fontWeight: FontWeight.w600),
      CellKind.conflict => const TextStyle(color: Palette.error, fontFamily: 'serif'),
      CellKind.doubtful => const TextStyle(color: Palette.doubt, fontFamily: 'serif'),
      _ => const TextStyle(color: Palette.ink, fontFamily: 'serif'),
    };
    BorderSide side(int index) => index == 8
        ? BorderSide.none
        : BorderSide(
            color: index % 3 == 2 ? Palette.box : Palette.line,
            width: index % 3 == 2 ? 2 : 1,
          );

    return Semantics(
      button: true,
      label: 'Ligne ${r + 1}, colonne ${c + 1}${value == 0 ? ', vide' : ', $value'}',
      child: InkWell(
        onTap: () => onTap(r, c),
        child: Container(
          decoration: BoxDecoration(
            color: background,
            border: Border(right: side(c), bottom: side(r)),
          ),
          alignment: Alignment.center,
          child: value == 0
              ? null
              : FittedBox(
                  child: Padding(
                    padding: const EdgeInsets.all(4),
                    child: Text('$value', style: style.copyWith(fontSize: 30)),
                  ),
                ),
        ),
      ),
    );
  }
}
