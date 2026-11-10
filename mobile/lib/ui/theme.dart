import 'package:flutter/material.dart';

/// Couleurs reprises du site web.
abstract final class Palette {
  static const ink = Color(0xFF1C2230);
  static const muted = Color(0xFF5D6578);
  static const paper = Color(0xFFF6F7FA);
  static const line = Color(0xFFCDD3DF);
  static const box = Color(0xFF4A5370);
  static const cobalt = Color(0xFF2445D4);
  static const cobaltSoft = Color(0xFFE8ECFB);
  static const error = Color(0xFFC2362B);
  static const errorSoft = Color(0xFFFBE9E7);
  static const doubt = Color(0xFFB26A00);
  static const doubtSoft = Color(0xFFFFF1D6);
}

/// Thème lisible : grands caractères, gros boutons, contrastes forts.
ThemeData buildTheme() {
  final base = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: Palette.cobalt,
      primary: Palette.cobalt,
      surface: Palette.paper,
    ),
    scaffoldBackgroundColor: Palette.paper,
  );
  return base.copyWith(
    textTheme: base.textTheme.apply(bodyColor: Palette.ink, displayColor: Palette.ink).copyWith(
          headlineMedium: const TextStyle(
            fontFamily: 'serif',
            fontSize: 34,
            fontWeight: FontWeight.w600,
            color: Palette.ink,
          ),
          bodyLarge: const TextStyle(fontSize: 20, height: 1.4, color: Palette.ink),
          bodyMedium: const TextStyle(fontSize: 18, height: 1.4, color: Palette.ink),
        ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(64),
        textStyle: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(60),
        foregroundColor: Palette.ink,
        side: const BorderSide(color: Palette.box, width: 1.5),
        textStyle: const TextStyle(fontSize: 19, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
  );
}
