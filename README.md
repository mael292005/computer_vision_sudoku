# Sudoku Vision

[![CI](https://github.com/mael292005/computer_vision_sudoku/actions/workflows/ci.yml/badge.svg)](https://github.com/mael292005/computer_vision_sudoku/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Un bot de computer vision qui regarde l'écran, lit une grille de Sudoku sur un site web, la résout et la remplit tout seul.**

<!-- Le GIF de démo arrivera quand le pipeline complet fonctionnera :
![Démo](docs/images/demo.gif)
-->

> 🚧 Projet en cours : voir la [feuille de route](#feuille-de-route) plus bas.

## Fonctionnement

Le bot est une chaîne de six étapes, chacune dans son propre module :

```
 Capture d'écran ──▶ Détection grille ──▶ Découpe des cases
  capture.py           grid.py              cells.py
                                                │
                                                ▼
 Saisie auto     ◀── Solveur          ◀── Lecture des chiffres
  actuator.py         solver.py             digits.py
```

| Étape | Rôle | Techniques principales |
| --- | --- | --- |
| 1. Capture | Capture l'écran (ou charge une image) | `mss` |
| 2. Détection de la grille | Trouve la grille et la redresse en un carré vu de face | Seuillage adaptatif, détection de contours, transformation de perspective (OpenCV) |
| 3. Découpe des cases | Découpe la grille en 81 cases et repère les cases vides | Découpage d'image, proportion de pixels « encre » |
| 4. Lecture des chiffres | Reconnaît le chiffre de chaque case pleine | Petit CNN (PyTorch), OCR Tesseract comme référence |
| 5. Solveur | Résout la grille 9×9 | Backtracking avec propagation de contraintes |
| 6. Saisie automatique | Clique sur chaque case vide et tape la réponse | `pyautogui` |

Les étapes 2 à 4 sont la vraie computer vision : elles transforment des pixels bruts en une grille 9×9 d'entiers que le solveur sait traiter.

## Installation

```bash
git clone https://github.com/mael292005/computer_vision_sudoku.git
cd computer_vision_sudoku
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Dépendances optionnelles, à installer au fil des étapes :

| Option | Installe | Sert à |
| --- | --- | --- |
| `screen` | `mss`, `pyautogui` | Capture d'écran et saisie automatique |
| `ml` | `torch`, `torchvision` | Entraînement et utilisation du CNN |
| `ocr` | `pytesseract` | Référence Tesseract (nécessite aussi le binaire Tesseract) |

## Utilisation

```bash
sudoku-vision version
```

Les autres commandes (`solve --image`, `solve --screen`, `--debug`) arriveront avec le pipeline.

## Structure du projet

```
src/sudoku_vision/   # le package : un module par étape du pipeline
tests/               # tests unitaires et d'intégration, images de test dans tests/data/
training/            # scripts d'entraînement du modèle de chiffres
models/              # poids du modèle entraîné
notebooks/           # exploration visuelle de chaque étape
docs/images/         # figures et GIF de démo utilisés dans ce README
```

## Développement

```bash
pre-commit install          # lint, formatage et typage à chaque commit
ruff check . && ruff format --check .
mypy
pytest --cov
```

Chaque push relance les mêmes vérifications dans GitHub Actions, sur Python 3.11 et 3.12.

## Feuille de route

- [x] Mise en place du projet : packaging, lint, typage, tests, CI
- [ ] Jeu de test : captures de grilles avec les chiffres attendus
- [ ] Solveur
- [ ] Détection de la grille et correction de perspective
- [ ] Découpe des cases et détection des cases vides
- [ ] Lecture des chiffres (comparaison CNN / Tesseract)
- [ ] Capture d'écran et saisie automatique
- [ ] CLI complète avec sortie `--debug`
- [ ] GIF de démo, métriques et bilan technique

**Plus tard :** photo d'un Sudoku papier, mode webcam en temps réel avec réalité augmentée, chiffres manuscrits.

## Licence

[MIT](LICENSE)
