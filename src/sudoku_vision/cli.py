"""Point d'entrée en ligne de commande : ``sudoku-vision``."""

from __future__ import annotations

import typer

from sudoku_vision import __version__

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


if __name__ == "__main__":  # pragma: no cover
    app()
