"""Polices utilisées pour générer les chiffres synthétiques du CNN.

Le CNN ne doit dépendre d'aucun site en particulier : on l'entraîne sur un
large échantillon de polices courantes du web, avec et sans empattements,
téléchargées depuis le dépôt officiel Google Fonts (licences libres OFL,
Apache et UFL). Les polices variables couvrent en plus toute une gamme de
graisses, du trait fin au trait gras.

La liste a été choisie pour couvrir les familles typiques des interfaces web,
pas pour imiter un site du jeu de test : les captures de tests/data ne servent
qu'à l'évaluation.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/"

GOOGLE_FONTS: tuple[str, ...] = (
    # Sans empattements
    "ofl/roboto/Roboto[wdth,wght].ttf",
    "ofl/opensans/OpenSans[wdth,wght].ttf",
    "ofl/notosans/NotoSans[wdth,wght].ttf",
    "ofl/lato/Lato-Light.ttf",
    "ofl/lato/Lato-Regular.ttf",
    "ofl/montserrat/Montserrat[wght].ttf",
    "ofl/nunito/Nunito[wght].ttf",
    "ofl/raleway/Raleway[wght].ttf",
    "ofl/worksans/WorkSans[wght].ttf",
    "ofl/rubik/Rubik[wght].ttf",
    "ofl/quicksand/Quicksand[wght].ttf",
    "ofl/mulish/Mulish[wght].ttf",
    "ofl/sourcesans3/SourceSans3[wght].ttf",
    "ofl/firasans/FiraSans-Light.ttf",
    "ofl/firasans/FiraSans-Regular.ttf",
    "ofl/ptsans/PT_Sans-Web-Regular.ttf",
    "ofl/karla/Karla[wght].ttf",
    "ofl/dmsans/DMSans[opsz,wght].ttf",
    "ofl/josefinsans/JosefinSans[wght].ttf",
    "ofl/comfortaa/Comfortaa[wght].ttf",
    "ofl/oswald/Oswald[wght].ttf",
    "ofl/plusjakartasans/PlusJakartaSans[wght].ttf",
    "ofl/lexend/Lexend[wght].ttf",
    "ofl/barlow/Barlow-Light.ttf",
    "ofl/barlow/Barlow-Regular.ttf",
    "ofl/hind/Hind-Regular.ttf",
    "ofl/cabin/Cabin[wdth,wght].ttf",
    "ofl/exo2/Exo2[wght].ttf",
    "ofl/titilliumweb/TitilliumWeb-Regular.ttf",
    "ufl/ubuntu/Ubuntu-Light.ttf",
    "ufl/ubuntu/Ubuntu-Regular.ttf",
    # Avec empattements
    "ofl/merriweather/Merriweather[opsz,wdth,wght].ttf",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/ptserif/PT_Serif-Web-Regular.ttf",
    "ofl/crimsontext/CrimsonText-Regular.ttf",
    "ofl/sourceserif4/SourceSerif4[opsz,wght].ttf",
    "ofl/notoserif/NotoSerif[wdth,wght].ttf",
    "ofl/ebgaramond/EBGaramond[wght].ttf",
)


def download_fonts(directory: Path = FONTS_DIR) -> list[Path]:
    """Télécharge les polices Google Fonts (une seule fois) et renvoie leurs chemins."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for remote in GOOGLE_FONTS:
        local = directory / Path(remote).name
        if not local.is_file():
            url = BASE_URL + urllib.parse.quote(remote)
            with urllib.request.urlopen(url, timeout=60) as response:
                local.write_bytes(response.read())
            print(f"  téléchargée : {local.name}")
        paths.append(local)
    return paths
