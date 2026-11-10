"""Point d'entrée en ligne de commande : ``sudoku-vision``."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from sudoku_vision import __version__, solver

app = typer.Typer(
    name="sudoku-vision",
    help="Lit une grille de Sudoku à l'écran, la résout et la remplit.",
    no_args_is_help=True,
    add_completion=False,
)

ModelOption = typer.Option(
    "cnn", "--model", "-m", help="Classifieur des chiffres : cnn ou template."
)
DebugOption = typer.Option(
    None, "--debug", "-d", help="Dossier où enregistrer les images de chaque étape."
)


def _fail(message: str) -> None:
    typer.echo(f"Erreur : {message}", err=True)
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """Interface en ligne de commande de Sudoku Vision."""


@app.command()
def version() -> None:
    """Affiche la version installée."""
    typer.echo(f"sudoku-vision {__version__}")


@app.command()
def solve(
    grid: str | None = typer.Option(
        None, "--grid", "-g", help="Grille de 81 caractères, 0 ou . pour les cases vides."
    ),
    image: Path | None = typer.Option(
        None, "--image", "-i", help="Capture d'écran contenant une grille."
    ),
    model: str = ModelOption,
    debug: Path | None = DebugOption,
) -> None:
    """Résout une grille donnée en texte (--grid) ou dans une image (--image)."""
    if (grid is None) == (image is None):
        _fail("donne soit --grid, soit --image")

    if grid is not None:
        try:
            solution = solver.solve(solver.parse_grid(grid))
        except (solver.InvalidGridError, solver.UnsolvableGridError) as error:
            _fail(str(error))
        typer.echo(solver.format_grid(solution))
        return

    from sudoku_vision.digits import load_classifier
    from sudoku_vision.grid import GridNotFoundError, load_image
    from sudoku_vision.pipeline import ReadingError, run, save_debug

    assert image is not None
    try:
        picture = load_image(image)
        result = run(picture, load_classifier(model))
    except (FileNotFoundError, GridNotFoundError, ReadingError) as error:
        _fail(str(error))

    typer.echo("Grille lue :")
    typer.echo(solver.format_grid(result.puzzle))
    typer.echo("\nSolution :")
    typer.echo(solver.format_grid(result.solution))
    steps = ", ".join(f"{name} {ms:.0f} ms" for name, ms in result.timings.items())
    typer.echo(f"\nTemps : {steps}")
    if debug is not None:
        for path in save_debug(result, picture, debug):
            typer.echo(f"Image de debug : {path}")


@app.command()
def play(
    model: str = ModelOption,
    delay: float = typer.Option(
        3.0, "--delay", help="Secondes d'attente avant la capture, pour afficher le site."
    ),
    fill: bool = typer.Option(
        True, "--fill/--no-fill", help="Remplir la grille ou seulement afficher la solution."
    ),
    debug: Path | None = DebugOption,
) -> None:  # pragma: no cover - nécessite un écran, une souris et un clavier
    """Capture l'écran, lit la grille, la résout et la remplit automatiquement.

    Ouvre une grille sur websudoku.com, lance la commande, puis affiche le site.
    Pour arrêter le bot à tout moment : amène la souris dans un coin de l'écran.
    """
    from sudoku_vision.actuator import PyAutoGuiController, fill_grid
    from sudoku_vision.capture import grab_screen
    from sudoku_vision.digits import load_classifier
    from sudoku_vision.grid import GridNotFoundError
    from sudoku_vision.pipeline import ReadingError, run, save_debug

    classifier = load_classifier(model)
    typer.echo(f"Capture dans {delay:.0f} s : affiche la grille à l'écran...")
    time.sleep(delay)

    shot = grab_screen()
    try:
        result = run(shot.image, classifier)
    except (GridNotFoundError, ReadingError) as error:
        _fail(str(error))
    if debug is not None:
        save_debug(result, shot.image, debug)

    typer.echo("Grille lue :")
    typer.echo(solver.format_grid(result.puzzle))
    typer.echo("\nSolution :")
    typer.echo(solver.format_grid(result.solution))
    if not fill:
        return

    start = time.perf_counter()
    count = fill_grid(
        result.puzzle, result.solution, shot.to_screen(result.centers), PyAutoGuiController()
    )
    typer.echo(f"\n{count} cases remplies en {time.perf_counter() - start:.1f} s.")


@app.command()
def benchmark(
    data: Path = typer.Option(
        Path("tests/data"), "--data", help="Dossier avec les images et grids.json."
    ),
    model: list[str] = typer.Option(
        ["template", "cnn"], "--model", "-m", help="Classifieur(s) à évaluer."
    ),
    scale: float = typer.Option(
        1.0, "--scale", help="Réduit les images (0.5 = moitié) pour tester la robustesse."
    ),
    blur: float = typer.Option(0.0, "--blur", help="Ajoute un flou gaussien (en pixels)."),
) -> None:
    """Mesure la précision de chaque étape sur un jeu de test annoté."""
    from sudoku_vision.digits import load_classifier
    from sudoku_vision.evaluate import evaluate, load_annotations

    try:
        samples = load_annotations(data)
    except FileNotFoundError as error:
        _fail(str(error))
    for index, name in enumerate(model):
        if index:
            typer.echo("")
        report = evaluate(samples, load_classifier(name), scale=scale, blur=blur)
        for line in report.lines():
            typer.echo(line)


@app.command()
def web(
    port: int = typer.Option(8000, "--port", "-p", help="Port du serveur local."),
    browser: bool = typer.Option(
        True, "--browser/--no-browser", help="Ouvrir le site dans le navigateur."
    ),
) -> None:  # pragma: no cover - lance un serveur qui tourne jusqu'à Ctrl+C
    """Lance le site web en local : saisie de grille ou capture, solution instantanée."""
    from sudoku_vision.webapp import WEB_DIR, prepare_site, serve

    try:
        prepare_site()
    except FileNotFoundError as error:
        _fail(str(error))
    serve(WEB_DIR, port, open_browser=browser)


if __name__ == "__main__":  # pragma: no cover
    app()
