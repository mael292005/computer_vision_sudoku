
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import '../core/cnn.dart';
import '../core/grid_detector.dart';
import '../core/pipeline.dart';
import 'result_screen.dart';
import 'theme.dart';

/// Lecture dans un isolate séparé, pour que l'écran reste fluide.
GridReading _readInBackground((Uint8List, ByteData) input) {
  final (photo, weights) = input;
  return readPhoto(photo, DigitCnn.fromBytes(weights));
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _picker = ImagePicker();
  bool _busy = false;
  String? _error;

  Future<void> _pick(ImageSource source) async {
    final file = await _picker.pickImage(
      source: source,
      maxWidth: 2000,
      maxHeight: 2000,
      imageQuality: 95,
    );
    if (file == null) return;

    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final photo = await file.readAsBytes();
      final weights = await rootBundle.load('assets/digits_cnn.bin');
      final reading = await compute(_readInBackground, (photo, weights));
      if (!mounted) return;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(builder: (_) => ResultScreen(reading: reading, photo: photo)),
      );
    } on GridNotFoundError {
      setState(() => _error =
          "Je n'ai pas trouvé de grille sur la photo. Prenez la grille entière, bien à plat, "
          'avec un bon éclairage, puis réessayez.');
    } catch (_) {
      setState(() => _error = "La photo n'a pas pu être lue. Essayez avec une autre photo.");
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 40, 24, 32),
          children: [
            Text('Sudoku Photo', style: text.headlineMedium),
            const SizedBox(height: 12),
            Text(
              'Photographiez une grille de Sudoku de votre journal ou de votre livre : '
              "l'application la lit et vous aide à la terminer.",
              style: text.bodyLarge,
            ),
            const SizedBox(height: 36),
            FilledButton.icon(
              onPressed: _busy ? null : () => _pick(ImageSource.camera),
              icon: const Icon(Icons.photo_camera_outlined, size: 28),
              label: const Text('Prendre la grille en photo'),
            ),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: _busy ? null : () => _pick(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined, size: 26),
              label: const Text('Choisir une photo'),
            ),
            const SizedBox(height: 28),
            if (_busy)
              const Row(
                children: [
                  SizedBox(
                    width: 28,
                    height: 28,
                    child: CircularProgressIndicator(strokeWidth: 3),
                  ),
                  SizedBox(width: 16),
                  Expanded(child: Text('Lecture de la grille…', style: TextStyle(fontSize: 20))),
                ],
              ),
            if (_error != null)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Palette.errorSoft,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(_error!, style: text.bodyLarge?.copyWith(color: Palette.error)),
              ),
            const SizedBox(height: 28),
            Text('Pour une bonne photo', style: text.titleLarge),
            const SizedBox(height: 8),
            for (final tip in const [
              'La grille entière dans la photo, avec un peu de marge autour.',
              'La page bien à plat, sans ombre de la main ou du téléphone.',
              'Une lumière franche, sans reflet sur le papier.',
            ])
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(top: 9, right: 12),
                      child: Icon(Icons.circle, size: 8, color: Palette.muted),
                    ),
                    Expanded(child: Text(tip, style: text.bodyMedium)),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
