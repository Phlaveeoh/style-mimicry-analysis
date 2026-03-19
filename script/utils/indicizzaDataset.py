import os
import pandas as pd

def indicizza_dataset_completo(percorso_base):
    data = []
    percorso_base = os.path.normpath(percorso_base)
    print(f"Indicizzazione avviata su: {percorso_base}")

    estensioni_valide = ('.png', '.jpg', '.jpeg')

    for root, dirs, files in os.walk(percorso_base):
        for file in files:
            if not file.lower().endswith(estensioni_valide):
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
                continue

            # Inizializzazione variabili
            macro_categoria = "Sconosciuta"
            sotto_categoria = "none"
            protezione = "none"
            metodo = "none"

            path_up = parts[:indice_artista]
            
            # Parsing basato sulla struttura documentata
            if "training_images" in path_up:
                macro_categoria = "Training"
                if "original" in path_up:
                    sotto_categoria = "Original"
                elif "protected+preprocessed" in path_up:
                    sotto_categoria = "Protected+Preprocessed"
                    metodo = parts[indice_artista - 1]
                    protezione = parts[indice_artista - 2]
                elif "protected" in path_up:
                    sotto_categoria = "Protected"
                    protezione = parts[indice_artista - 1]
                    
            elif "generated_images" in path_up:
                macro_categoria = "Generated"
                if "no-protections" in path_up:
                    sotto_categoria = "No-Protections"
                    metodo = parts[indice_artista - 1]
                    protezione = parts[indice_artista - 2]
                elif "naive-mimicry" in path_up:
                    sotto_categoria = "Naive-Mimicry"
                    protezione = parts[indice_artista - 1]
                else:
                    sotto_categoria = "Robust-Mimicry"
                    metodo = parts[indice_artista - 1]
                    protezione = parts[indice_artista - 2]

            # Aggiunta record
            data.append({
                "path": full_path,
                "filename": file,
                "artista": nome_artista,
                "categoria": macro_categoria,
                "tipo": sotto_categoria,
                "protezione": protezione,
                "metodo_processamento": metodo,
            })

    df = pd.DataFrame(data)
    print(f"Indicizzazione completata. Trovate {len(df)} immagini.")
    return df