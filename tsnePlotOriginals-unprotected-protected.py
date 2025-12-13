import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.models import load_model
from keras.utils import load_img, img_to_array

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_pre
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_pre
from keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as effnet_pre
from keras.applications.convnext import ConvNeXtBase, preprocess_input as convnext_pre

from estraiLayer import estraiLayerStile, estraiUltimoLayer 
from indicizzaDataset import indicizza_dataset_completo

mappa_funzioni = {
    "inceptionresnetv2": inception_pre,
    "resnet50": resnet_pre,
    "efficientnetv2m": effnet_pre,
    "convnextbase": convnext_pre,
    "functional4": convnext_pre
}

percorso_modello = "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras"
input_size = (224, 224)
percorso_dataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"

protezione = "antidb"
preprocess = "noisy_upscaling"

artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
    ]

#Caricamento dataset
df = indicizza_dataset_completo(percorso_dataset)

if df.empty:
    print("dataset vuoto.")
    exit()

#Gruppo A: Immagini originali
df_originals = df[
    (df["artista"].isin(artisti)) &
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy()

print(len(df_originals))

df_nm_unprotected = df[
    (df["artista"].isin(artisti)) &
    (df["categoria"] == "Generated") &
    (df["tipo"] == "No-Protections") &
    (df["protezione"] == protezione) &
    (df["metodo_processamento"] == preprocess)
].copy()

print(len(df_nm_unprotected))
df_nm_unprotected["artista"] = df_nm_unprotected["artista"] + "_nm_unprotected"

# --- GRUPPO C: Generated Protected ---
df_nm_protected = df[
    (df["artista"].isin(artisti)) &
    (df["categoria"] == "Generated") &
    (df["tipo"] == "Naive-Mimicry") &
    (df["protezione"] == protezione)
].copy()
print(len(df_nm_protected))
df_nm_protected["artista"] = df_nm_protected["artista"] + "_nm_protected"

df_plot = pd.concat([df_originals, df_nm_unprotected, df_nm_protected], ignore_index=True)

#Carico il modello fine-tuned
try:
    modello_base = load_model(percorso_modello)
    #modello_base = ConvNeXtBase(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    nome_modello = modello_base.name.lower().replace("-", "").replace("_", "")
    preprocess_input = mappa_funzioni[nome_modello]
    layer_stile = estraiUltimoLayer(modello_base)
    print("Modello caricato correttamente.")
except Exception as e:
    print(f"Errore modello: {e}")
    exit()

#Estrazione feature
feature_list = []
labels_list = []

print(f"\nEstrazione feature ({len(df_plot)} immagini)...")
for i, row in df_plot.iterrows():
    try:
        img = load_img(row["path"], target_size=input_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        pred = layer_stile.predict(x).flatten()
        feature_list.append(pred)
        labels_list.append(row["artista"])
    except Exception as e:
        print(f"\nErrore su {row['filename']}: {e}")

if len(feature_list) == 0:
    print("\nERRORE: Nessuna feature estratta.")
    exit()

#Converto le feature in array numpy per t-SNE
X = np.array(feature_list)

print(f"\nCalcolo t-SNE su {len(X)} punti...")

#Perplexity in base al numero di campioni
if len(df_plot) < 100:
    p = 10
elif 100 <= len(df_plot) <= 500:
    p = 30
elif 501 <= len(df_plot) <= 1000:
    p = 50
else:
    p = 100
    
print(f"Con perplexity = {p}")
tsne = TSNE(n_components=2, random_state=42, perplexity=p)

Y = tsne.fit_transform(X)

#plot del grafico
plt.figure(figsize=(14, 10))

palette = {
    # --- 1. Albrecht Durer (Famiglia dei ROSSI) ---
    "wikiart_albrecht-durer":                 "#8B0000", # DarkRed (Originale - Scuro)
    "wikiart_albrecht-durer_nm_unprotected":  "#FF0000", # Red (Imitazione Riuscita - Vivido)
    "wikiart_albrecht-durer_nm_protected":    "#FD885A", # LightSalmon (Imitazione Protetta - Pastello)

    # --- 2. Alphonse Mucha (Famiglia dei BLU) ---
    "wikiart_alphonse-mucha":                 "#000080", # Navy (Originale - Molto Scuro)
    "wikiart_alphonse-mucha_nm_unprotected":  "#0000FF", # Blue (Imitazione Riuscita - Vivido)
    "wikiart_alphonse-mucha_nm_protected":    "#ADD8E6", # LightBlue (Imitazione Protetta - Pastello)

    # --- 3. Anna Ostroumova Lebedeva (Famiglia dei VIOLA) ---
    # Sostituito l'oro con il viola per leggibilità
    "wikiart_anna-ostroumova-lebedeva":                 "#4B0082", # Indigo (Originale - Scuro)
    "wikiart_anna-ostroumova-lebedeva_nm_unprotected":  "#9400D3", # DarkViolet (Imitazione Riuscita - Vivido)
    "wikiart_anna-ostroumova-lebedeva_nm_protected":    "#A75FA7", # Thistle (Imitazione Protetta - Pastello)

    # --- 4. Edvard Munch (Famiglia degli ARANCIONI) ---
    # Sostituito il Ciano con l'Arancione per massimo contrasto col Blu
    "wikiart_edvard-munch":                 "#8B4500", # SaddleBrown (Originale - Scuro/Marrone)
    "wikiart_edvard-munch_nm_unprotected":  "#DA7800", # DarkOrange (Imitazione Riuscita - Vivido)
    "wikiart_edvard-munch_nm_protected":    "#F5A259", # PeachPuff (Imitazione Protetta - Pastello)

    # --- 5. Edward Hopper (Famiglia dei VERDI) ---
    "wikiart_edward-hopper":                 "#006400", # DarkGreen (Originale - Scuro)
    "wikiart_edward-hopper_nm_unprotected":  "#1BB11B", # LimeGreen (Imitazione Riuscita - Vivido)
    "wikiart_edward-hopper_nm_protected":    "#6EF16E", # PaleGreen (Imitazione Protetta - Pastello)
}

gruppi = df_plot["artista"].unique()

for gruppo in gruppi:
    # Trova gli indici corrispondenti a questo gruppo
    indici = [i for i, lab in enumerate(labels_list) if lab == gruppo]
    
    if not indici:
        continue

    plt.scatter( Y[indici, 0], Y[indici, 1], c=palette.get(gruppo, "gray"), label=gruppo )

plt.title(f"Originali vs Naive Mimicry {protezione} vs Naive Mimicry Unprotected", fontsize=14, fontweight='bold')
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    borderaxespad=0.,
    fontsize=12
)
plt.tight_layout()
plt.show()