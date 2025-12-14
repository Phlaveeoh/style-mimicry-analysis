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

def indicizza_dataset_completo(percorso_base):
    data = []
    percorso_base = os.path.normpath(percorso_base)
    
    #macro categoria
    generate = "generated_images"
    training = "training_images"
    
    print(f"Indicizzazione avviata su: {percorso_base}")

    for root, dirs, files in os.walk(percorso_base):
        for file in files:
            # Filtriamo solo le immagini
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            full_path = os.path.join(root, file)
            parts = full_path.split(os.sep)
            
            indice_artista = -1
            nome_artista = None
            
            for i, part in enumerate(parts):
                if part.startswith("wikiart_"):
                    nome_artista = part
                    indice_artista = i
                    break
            
            if indice_artista == -1:
                continue # Salta file di sistema o fuori struttura

            #Variabili base
            macro_categoria = ""            # Generated vs Training
            sotto_categoria = "standard"    # Original vs Protected vs Preprocessed
            protezione = "none"             # antidb, glaze, mist
            metodo = "none"                 # diffpure, impress++, naive, noisy_upscaling

            #Risalgo la cartella a partire dall'artista per capire che tipo di immagini contiene
            path_up = parts[:indice_artista]
            
            if training in path_up:
                macro_categoria = "Training"
                
                if "original" in path_up:
                    sotto_categoria = "Original"
                    
                elif "protected+preprocessed" in path_up:
                    sotto_categoria = "Protected+Preprocessed"
                    # Struttura: .../protected+preprocessed/PROTEZIONE/METODO/artista
                    protezione = parts[indice_artista - 2]
                    metodo = parts[indice_artista - 1]
                    
                elif "protected" in path_up:
                    sotto_categoria = "Protected"
                    # Struttura: .../protected/PROTEZIONE/artista
                    protezione = parts[indice_artista - 1]

            # B. RAMO GENERATED IMAGES
            elif generate in path_up:
                macro_categoria = "Generated"
                sotto_categoria = "Robust-Mimicry"
                
                # --- NUOVO BLOCCO: NO-PROTECTIONS (Immagini "Clean") ---
                # Percorso tipo: .../no-protections/mist/noisy_upscaling/wikiart_artista
                if "no-protections" in path_up:
                    sotto_categoria = "No-Protections"
                    
                    metodo = parts[indice_artista - 1] 
                    # NOTA: Qui indica il GRUPPO DI CONTROLLO, non che l'immagine è protetta.
                    protezione = parts[indice_artista - 2]
                    
                # Struttura: .../naive-mimicry/PROTEZIONE/artista
                #Immagini generate senza metodi particolari da immagini protette
                if "naive-mimicry" in path_up:
                    sotto_categoria = "Naive-Mimicry"
                    protezione = parts[indice_artista - 1]
                else:
                    # Struttura Standard: .../PROTEZIONE/METODO/artista
                    #Immagini generate con metodi specifici da immagini protette
                    metodo = parts[indice_artista - 1]
                    protezione = parts[indice_artista - 2]
            
            '''       
            # --- 4. ANALISI DEL PERCORSO (SOTTO L'ARTISTA) ---
            # Verifica se c'è qualcosa tra l'artista e il file (es. 'train', 'val', '0.5')
            if len(parts) - 1 > indice_artista:
                sub_folder = parts[indice_artista + 1]
                
                # Distinguiamo tra SPLIT (train/val) e PARAMETRI (0.5, etc)
                if sub_folder.lower() in ["train", "val", "test"]:
                    split = sub_folder.lower()
                elif sub_folder != file: # Se non è il file stesso
                    parametro = sub_folder '''

            # --- 5. AGGIUNTA AL DATASET ---
            data.append({
                "path": full_path,
                "filename": file,
                "artista": nome_artista,
                "categoria": macro_categoria, # Training o Generated
                "tipo": sotto_categoria,      # Original, Protected, Mimicry...
                "protezione": protezione, # antidb, glaze, mist
                "metodo_processamento": metodo,    # diffpure, impress++, noisy_upscaling
            })

    df = pd.DataFrame(data)
    print(f"Indicizzazione completata. Trovate {len(df)} immagini.")
    return df

"""
percorso = "C:\\Users\\Flavio\\Desktop\\datasetTotale" # Il tuo percorso
df = indicizza_dataset_completo(percorso)

nome_file_csv = "dataset_immagini_completo.csv"
df.to_csv(nome_file_csv, index=False)

print(f"Dataset salvato correttamente in: {nome_file_csv}")
"""