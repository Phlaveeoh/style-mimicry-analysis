import os
from pathlib import Path
import pandas as pd

def indicizzaDataset(percorso_base):
    base = Path(percorso_base)
    righe = []

    # Sezione protette
    for protezione_dir in (base / "protected").iterdir():
        if protezione_dir.is_dir():
            protezione = protezione_dir.name
            for artista_dir in protezione_dir.iterdir():
                if artista_dir.is_dir():
                    artista = artista_dir.name
                    for img in artista_dir.glob("*.*"):
                        righe.append({
                            "path": str(img),
                            "stato": "protetta",
                            "protezione": protezione,
                            "artista": artista
                        })

    # Sezione non protette
    for artista_dir in (base / "original").iterdir():
        if artista_dir.is_dir():
            artista = artista_dir.name
            for img in artista_dir.glob("*.*"):
                righe.append({
                    "path": str(img),
                    "stato": "originale",
                    "protezione": None,
                    "artista": artista
                })

    return pd.DataFrame(righe)

