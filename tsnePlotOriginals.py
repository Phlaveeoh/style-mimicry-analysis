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
    "convnextbase": convnext_pre
}

percorso_modello = "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras"
input_size = (224, 224)
percorso_dataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"

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

#Filtro immagini originali
df_plot = df[
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy()

#Carico il modello fine-tuned
try:
    modello_base = load_model(percorso_modello)
    #modello_base = ConvNeXtBase(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    #layer_stile = estraiLayerStile(modello_base)
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
        x = convnext_pre(x)
        
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

tsne = TSNE(n_components=2, random_state=42, perplexity=10)

Y = tsne.fit_transform(X)

#plot del grafico
plt.figure(figsize=(12, 8))

# Mappa colori
palette = {
    "wikiart_albrecht-durer": "red",
    "wikiart_alphonse-mucha": "blue",
    "wikiart_anna-ostroumova-lebedeva": "yellow",
    "wikiart_edvard-munch": "cyan",
    "wikiart_edward-hopper": "green",
}

gruppi = df_plot["artista"].unique()

for gruppo in gruppi:
    # Trova gli indici corrispondenti a questo gruppo
    indici = [i for i, lab in enumerate(labels_list) if lab == gruppo]
    
    if not indici:
        continue

    plt.scatter( Y[indici, 0], Y[indici, 1], c=palette.get(gruppo, "gray"), label=gruppo )

plt.title(f"T-SNE su modello fine tuning per massimizzare la distanza tra gli artisti", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()