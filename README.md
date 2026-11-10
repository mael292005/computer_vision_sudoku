# Sudoku Vision

[![CI](https://github.com/mael292005/computer_vision_sudoku/actions/workflows/ci.yml/badge.svg)](https://github.com/mael292005/computer_vision_sudoku/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Licence](https://img.shields.io/badge/licence-MIT-green)

**Un bot de computer vision qui regarde l'écran, lit une grille de Sudoku sur un site web, la résout et la remplit tout seul.**

Site cible : [websudoku.com](https://www.websudoku.com/).

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

### Les étapes de vision en images

![Étapes de vision](docs/images/pipeline_vision.png)

1. **Détection** : après un seuillage, on garde le plus grand contour à 4 coins presque carré. Le logo « WEB SUDOKU », plus petit, est écarté.
2. **Seuillage adaptatif** : chaque pixel est comparé à son voisinage, ce qui isole les traits et les chiffres.
3. **Redressement et découpe** : la grille est projetée en un carré de 450 px, puis découpée en 81 cases. Une case est pleine (en vert) si plus de 1 % de ses pixels sont sombres.

## Jeu de test

Quatre captures d'écran complètes de websudoku.com, une par niveau (Easy, Medium, Hard, Evil), sont dans `tests/data/`. Les chiffres attendus sont notés dans `tests/data/grids.json`. Chaque grille annotée a exactement une solution, ce qui confirme que la recopie est juste.

| Étape | Résultat sur le jeu de test |
| --- | --- |
| Détection de la grille | 4 / 4 grilles trouvées |
| Cases vides ou pleines | 324 / 324 cases correctes |
| Lecture des chiffres | à venir |

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

Résoudre une grille donnée sous forme de texte (81 caractères, `0` ou `.` pour les cases vides) :

```bash
sudoku-vision solve --grid "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."
```

```
8 1 2 | 7 5 3 | 6 4 9
9 4 3 | 6 8 2 | 1 7 5
6 7 5 | 4 9 1 | 2 8 3
------+-------+------
1 5 4 | 2 3 7 | 8 9 6
3 6 9 | 8 4 5 | 7 2 1
2 8 7 | 1 6 9 | 5 3 4
------+-------+------
5 2 1 | 9 7 4 | 3 6 8
4 3 8 | 5 2 6 | 9 1 7
7 9 6 | 3 1 8 | 4 5 2
```

Les commandes `solve --image`, `solve --screen` et l'option `--debug` arriveront avec le pipeline de vision.

## Structure du projet

```
src/sudoku_vision/   # le package : un module par étape du pipeline
tests/               # tests unitaires et d'intégration, images de test dans tests/data/
training/            # scripts d'entraînement du modèle de chiffres
models/              # poids du modèle entraîné
notebooks/           # exploration visuelle de chaque étape
scripts/             # génération des figures du README
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
- [x] Jeu de test : 4 captures (Easy à Evil) avec les chiffres attendus
- [x] Solveur : backtracking + heuristique MRV et masques de bits
- [x] Détection de la grille et correction de perspective
- [x] Découpe des cases et détection des cases vides
- [ ] Lecture des chiffres (comparaison CNN / Tesseract)
- [ ] Capture d'écran et saisie automatique
- [ ] CLI complète avec sortie `--debug`
- [ ] GIF de démo, métriques et bilan technique

**Plus tard :** photo d'un Sudoku papier, mode webcam en temps réel avec réalité augmentée, chiffres manuscrits.

## Licence

[MIT](LICENSE)
