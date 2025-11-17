from prendiImmagini import prendiImmagini
from estraiLayer import estraiUltimoLayer, estraiUltimoMultidimensionale
from indicizzaDataset import indicizzaDataset

from keras.applications.vgg19 import VGG19, preprocess_input as vgg_pre
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_pre
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_pre
from keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as effnet_pre
from keras.applications.convnext import ConvNeXtBase, preprocess_input as convnext_pre

from keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# dizionario che associa la classe del modello alla funzione di preprocessing corretta
mappa_funzioni = {
    "vgg19": vgg_pre,
    "inceptionresnetv2": inception_pre,
    "resnet50": resnet_pre,
    "efficientnetv2m": effnet_pre,
    "convnextbase": convnext_pre
}

#Carico un modello pre-addestrato
modello_base = ConvNeXtBase(weights='imagenet')

input_size = modello_base.input_shape[1:3]
nome_modello = modello_base.name.lower().replace("_", "")
preprocess_input = mappa_funzioni[nome_modello]
print(f"Modello: {modello_base.name} funzione preprocess_input corretta caricata.")

layer1 = estraiUltimoMultidimensionale(modello_base)
layer2 = estraiUltimoLayer(modello_base)
print(layer1.output.shape)
print(layer2.output.shape)

#Lista degli artisti scelti
artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
    ]

colori_artisti = {
    "wikiart_albrecht-durer": "red",
    "wikiart_alphonse-mucha": "blue",
    "wikiart_anna-ostroumova-lebedeva": "yellow",
    "wikiart_edvard-munch": "cyan",
    "wikiart_edward-hopper": "green",
}

protezione = "antidb"

feature1, feature2 = [], []
labels, segmenti = [], []
nomi_immagini = []

#Caricamento del Dataframe
percorsoDataset = "C:\\Users\\Flavio\\Desktop\\dataset"
dataframe = indicizzaDataset(percorsoDataset)

for artista in artisti:
    df_artista = dataframe[dataframe["artista"] == artista].copy()
    df_artista["filename"] = df_artista["path"].apply(lambda x: os.path.basename(x))
    df_artista["id_img"] = df_artista["filename"].str.replace(".png", "", regex=False)
    
    nonprotette = df_artista[df_artista["stato"] == "originale"]
    protette_antidb = df_artista[(df_artista["stato"] == "protetta") & (df_artista["protezione"] == protezione)]
    coppie = nonprotette.merge(protette_antidb, on="id_img", suffixes=("_nonprotetta", "_protetta"))

    paths_nonprotette = coppie["path_nonprotetta"].tolist()
    paths_protette = coppie["path_protetta"].tolist()

    print(f"Trovate {len(coppie)} coppie di immagini non protette/protette con antidb per l'artista {artista}")
    
    for _, row in coppie.iterrows():
        for stato in ["nonprotetta", "protetta"]:
            path = row[f"path_{stato}"]
            img = load_img(path, target_size=input_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            predizione1 = layer1.predict(x).flatten()
            predizione2 = layer2.predict(x).flatten()

            feature1.append(predizione1)
            feature2.append(predizione2)
            labels.append(artista)
            
            #salvo il nome dell'immagine
            nome_file = os.path.basename(path)
            nome_file = os.path.splitext(nome_file)[0]
            nome_file = nome_file.lstrip("0") or "0"
            if stato == "protetta":
                nome_file += "P"
            nomi_immagini.append(nome_file)
        # Salva coppia per tracciare segmento
        segmenti.append((len(feature1)-2, len(feature1)-1))

#TSNE
X1 = np.array(feature1)
X2 = np.array(feature2)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)

Y1 = tsne.fit_transform(X1)
Y2 = tsne.fit_transform(X2)

#plot
plt.figure(figsize=(16,8))

titoli = ["Layer multidimensionale", "Layer finale"]

for index, (Y, titolo) in enumerate(zip([Y1, Y2], titoli), start=1):
    plt.subplot(1,2,index)
    for artista in artisti:
        inds = [i for i, lab in enumerate(labels) if lab == artista]
        plt.scatter(Y[inds,0], Y[inds,1], c=colori_artisti[artista], label=artista)
        
    #annota ogni punto con il nome dell'immagine
    for i, nome in enumerate(nomi_immagini):
        plt.text(Y[i,0], Y[i,1], nome, fontsize=8)
    
    # Disegna segmenti tra immagini protette/non protette
    for i_nonprot, i_prot in segmenti:
        plt.plot([Y[i_nonprot,0], Y[i_prot,0]], [Y[i_nonprot,1], Y[i_prot,1]], 'k--', alpha=0.5)

    plt.title(f"{titolo}\nProtezione: {protezione}", fontsize=12, fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='lower left')
plt.tight_layout()
plt.show()