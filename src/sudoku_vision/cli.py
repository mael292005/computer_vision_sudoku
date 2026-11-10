"""Point d'entrée en ligne de commande : ``sudoku-vision``."""

from __future__ import annotations

import typer

from sudoku_vision import __version__, solver

app = typer.Typer(
    name="sudoku-vision",
    help="Lit une grille de Sudoku à l'écran, la résout et la remplit.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Interface en ligne de commande de Sudoku Vision."""


@app.command()
def version() -> None:
    """Affiche la version installée."""
    typer.echo(f"sudoku-vision {__version__}")


@app.command()
def solve(
    grid: str = typer.Option(
        ...,
        "--grid",
        "-g",
        help="Grille de 81 caractères, 0 ou . pour les cases vides.",
    ),
) -> None:
    """Résout une grille donnée sous forme de texte."""
    try:
        puzzle = solver.parse_grid(grid)
        solution = solver.solve(puzzle)
    except (solver.InvalidGridError, solver.UnsolvableGridError) as error:
        typer.echo(f"Erreur : {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(solver.format_grid(solution))


if __name__ == "__main__":  # pragma: no cover
    app()
