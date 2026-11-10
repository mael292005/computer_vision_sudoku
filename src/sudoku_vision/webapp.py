"""Lancement du site web en local : ``sudoku-vision web``.

Le site (dossier ``web/`` du dépôt) est entièrement statique : la vision tourne
dans le navigateur. Il suffit donc de copier les fichiers partagés avec la
version Python (le CNN et les captures d'exemple) puis de servir le dossier.
Un serveur est nécessaire : les navigateurs refusent de charger des modules
JavaScript et des fichiers depuis ``file://``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"

# Même liste que web/scripts/sync-assets.mjs : (source dans le dépôt, destination dans web/assets).
SHARED_FILES: tuple[tuple[str, str], ...] = (
    ("src/sudoku_vision/assets/digits_cnn.onnx", "digits_cnn.onnx"),
    ("tests/data/websudoku_easy.png", "examples/easy.png"),
    ("tests/data/websudoku_medium.png", "examples/medium.png"),
    ("tests/data/websudoku_hard.png", "examples/hard.png"),
    ("tests/data/websudoku_evil.png", "examples/evil.png"),
    ("tests/data/sudokucom_master.png", "examples/sudokucom.png"),
    ("tests/data/onlinesudoku_easy.png", "examples/onlinesudoku.png"),
)


def prepare_site(web_dir: Path = WEB_DIR, root: Path = REPO_ROOT) -> list[Path]:
    """Copie le modèle et les exemples dans ``web_dir/assets``.

    Raises:
        FileNotFoundError: si le site ou un fichier partagé est introuvable
            (le site n'existe que dans le dépôt, pas dans un paquet installé).

    Returns:
        Les fichiers copiés.
    """
    if not (web_dir / "index.html").is_file():
        raise FileNotFoundError(
            f"site introuvable dans {web_dir} : lancez la commande depuis le dépôt cloné"
        )
    copied = []
    for source, target in SHARED_FILES:
        destination = web_dir / "assets" / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source, destination)
        copied.append(destination)
    return copied


def serve(web_dir: Path, port: int, open_browser: bool) -> None:  # pragma: no cover
    """Sert le site sur http://localhost:<port> jusqu'à Ctrl+C."""
    import functools
    import http.server
    import webbrowser

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {  # noqa: RUF012 - surcharge de l'attribut de la classe parente
            **http.server.SimpleHTTPRequestHandler.extensions_map,
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".wasm": "application/wasm",
        }

        def log_message(self, format: str, *args: object) -> None:
            pass  # pas de ligne par requête dans le terminal

    handler = functools.partial(Handler, directory=str(web_dir))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        url = f"http://localhost:{port}/"
        print(f"Site disponible sur {url} (Ctrl+C pour arrêter)")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nSite arrêté.")
