import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.convnext import preprocess_input as convnext_pre

from estraiLayer import estraiLayerStile 
from indicizzaDataset import indicizza_dataset_completo

percorso_modello = "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_alphonse-mucha.keras"
input_size = (224, 224)
percorso_dataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"

artista_target = "wikiart_alphonse-mucha"
PROTEZIONE_TARGET = "mist"

#Caricamento dataset
df = indicizza_dataset_completo(percorso_dataset)

if df.empty:
    print("ERRORE: Il dataset indicizzato è vuoto.")
    exit()

#Filtro immagini originali
df_orig = df[
    (df["artista"] == artista_target) & 
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy()
df_orig["Label_Plot"] = "Immagini Originali"

#filtro immagini generate partendo da immagini protette
df_protected = df[
    (df["artista"] == artista_target) & 
    (df["categoria"] == "Generated") & 
    (df["protezione"] == PROTEZIONE_TARGET)
].copy()
df_protected["Label_Plot"] = f"Generate da {PROTEZIONE_TARGET}"

#unione del dataset
df_plot = pd.concat([df_orig, df_protected], ignore_index=True)

print(f"\n--- ANALISI PER: {PROTEZIONE_TARGET} ---")
print(df_plot["Label_Plot"].value_counts())

if df_plot.empty:
    print("Nessun dato trovato con i filtri correnti. Controlla 'artista_target' o i percorsi.")
    exit()

#Carico il modello fine-tuned
try:
    modello_base = load_model(percorso_modello)
    layer_stile = estraiLayerStile(modello_base)
    print("Modello caricato correttamente.")
except Exception as e:
    print(f"Errore modello: {e}")
    exit()

#Inizio ad estrarre le feature
feature_list = []
labels_list = []

print(f"\nEstrazione feature ({len(df_plot)} immagini)...")
for i, row in df_plot.iterrows():
    try:
        img = load_img(row["path"], target_size=input_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = convnext_pre(x)
        
        pred = layer_stile.predict(x).flatten()
        feature_list.append(pred)
        labels_list.append(row["Label_Plot"])
    except Exception as e:
        print(f"\nErrore su {row['filename']}: {e}")

if len(feature_list) == 0:
    print("\nERRORE: Nessuna feature estratta.")
    exit()

#Converto le feature in array numpy per t-SNE
X = np.array(feature_list)

print(f"\nCalcolo t-SNE su {len(X)} punti...")
# Perplexity basata sul numero di immagini nel dataset
perplex = min(30, len(X) - 1)

tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)

Y = tsne.fit_transform(X)

#plot del grafico
plt.figure(figsize=(12, 8))

# Mappa colori
palette = {
    "Immagini Originali": "blue",
    f"Generate da {PROTEZIONE_TARGET}": "red"
}

gruppi = df_plot["Label_Plot"].unique()

for gruppo in gruppi:
    # Trova gli indici corrispondenti a questo gruppo
    indici = [i for i, lab in enumerate(labels_list) if lab == gruppo]
    
    if not indici:
        continue

    plt.scatter( Y[indici, 0], Y[indici, 1], c=palette.get(gruppo, "gray"), label=gruppo )

plt.title(f"Analisi Efficacia Protezione: {PROTEZIONE_TARGET}\nArtista: {artista_target}", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()