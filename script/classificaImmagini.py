import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.convnext import preprocess_input as convnext_pre

from utils.indicizzaDataset import indicizza_dataset_completo

parser = argparse.ArgumentParser(description="Classifica immagini e genera report visivo.")
parser.add_argument("-m", "--modello", type=str, required=True, help="Percorso del modello")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Percorso del dataset")
parser.add_argument("-c", "--classes", type=str, required=True, help="Classi separate da virgola")
parser.add_argument("--artista", type=str, help="Filtro artista (es. wikiart_edward-hopper)")
parser.add_argument("--categoria", type=str, help="Filtro categoria (es. Training, Generated)")
parser.add_argument("--tipo", type=str, help="Filtro tipo (es. Original, Protected, Robust-Mimicry)")
parser.add_argument("--protezione", type=str, help="Filtro protezione (es. glaze, mist)")
parser.add_argument("--metodo", type=str, help="Filtro metodo_processamento (es. noisy_upscaling)")
args = parser.parse_args()

input_size = (224, 224)
class_names = [c.strip() for c in args.classes.split(',')]

df = indicizza_dataset_completo(args.dataset)
if df.empty:
    print("Errore: Dataset vuoto.")
    exit(1)

if args.artista:
    df = df[df["artista"] == args.artista]
if args.categoria:
    df = df[df["categoria"] == args.categoria]
if args.tipo:
    df = df[df["tipo"] == args.tipo]
if args.protezione:
    df = df[df["protezione"] == args.protezione]
if args.metodo:
    df = df[df["metodo_processamento"] == args.metodo]

if df.empty:
    print("Nessuna immagine trovata con i filtri forniti.")
    exit(1)

modello_base = load_model(args.modello)
risultati = []

for i, row in df.iterrows():
    img_raw = load_img(row["path"], target_size=input_size)
    x = img_to_array(img_raw)
    x = np.expand_dims(x, axis=0)
    x = convnext_pre(x)
    
    pred = modello_base.predict(x, verbose=0)[0]
    idx_pred = np.argmax(pred)
    classe_predetta = class_names[idx_pred] if idx_pred < len(class_names) else str(idx_pred)
    confidenza = pred[idx_pred]
    
    risultati.append({
        "path": row["path"],
        "reale": row["artista"],
        "predetta": classe_predetta,
        "confidenza": confidenza,
        "img_obj": img_raw 
    })

df_risultati = pd.DataFrame(risultati)

print(df_risultati[["path", "reale", "predetta", "confidenza"]].to_string(index=False))

num_img = len(df_risultati)
cols = min(5, num_img)
rows = int(np.ceil(num_img / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
if num_img == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for i, row in df_risultati.iterrows():
    ax = axes[i]
    ax.imshow(row["img_obj"])
    ax.axis('off')
    
    colore_testo = "green" if row["reale"] == row["predetta"] else "red"
    titolo = f"Real: {row['reale']}\nPred: {row['predetta']}\nConf: {row['confidenza']:.2f}"
    ax.set_title(titolo, color=colore_testo, fontsize=10)

for j in range(num_img, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
os.makedirs("classification_plots", exist_ok=True)

filtri = [f for f in [args.artista, args.categoria, args.tipo, args.protezione, args.metodo] if f]
nome_file = "grid_predictions_" + ("_".join(filtri) if filtri else "all") + ".png"
percorso_output = os.path.join("classification_plots", nome_file)

plt.savefig(percorso_output, dpi=300)
print(f"Plot salvato in: {percorso_output}")