import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from keras.models import load_model
from keras.utils import load_img, img_to_array

#Import per i modelli
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_pre
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet_pre
from keras.applications.efficientnet_v2 import EfficientNetV2M, preprocess_input as effnet_pre
from keras.applications.convnext import ConvNeXtBase, preprocess_input as convnext_pre

#Import utils personali
from utils.estraiLayer import estraiLayerStile, estraiUltimoLayer 
from utils.indicizzaDataset import indicizza_dataset_completo

#--- CONFIGURAZIONE ARGPARSE ---
parser = argparse.ArgumentParser(
    description="Crea un plot t-SNE dinamico confrontando gruppi di immagini definiti dall'utente."
)

parser.add_argument(
    "-m", "--modello",
    type=str,
    required=True,
    help="Percorso del modello da utilizzare per l'estrazione delle feature."
)

parser.add_argument(
    "-d", "--dataset",
    type=str,
    required=True,
    help="Percorso della cartella contenente il dataset."
)

parser.add_argument(
    "-g", "--group",
    action='append',
    required=True,
    help="Definisci un gruppo da plottare. Sintassi: 'EtichettaLegenda|colonna=valore,colonna2=valore'. "
         "Esempio: 'Originali|tipo=Original,categoria=Training'"
)

args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
mappa_funzioni = {
    "inceptionresnetv2": inception_pre,
    "resnet50": resnet_pre,
    "efficientnetv2m": effnet_pre,
    "convnextbase": convnext_pre,
    "functional4": convnext_pre
}

input_size = (224, 224)

artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
]

#--- CARICAMENTO E FILTRAGGIO DATASET ---
print(f"Indicizzazione dataset da: {args.dataset}")
df = indicizza_dataset_completo(args.dataset)

if df.empty:
    print("Dataset vuoto o non trovato.")
    exit()

#Filtro preliminare sugli artisti
df = df[df["artista"].isin(artisti)].copy()

df_plot_list = []
print(f"Generazione gruppi basata su {len(args.group)} definizioni...")

#Parsing dei gruppi dinamici
for group_def in args.group:
    try:
        #Divide nome gruppo dai filtri
        label_name, filters_str = group_def.split('|')
        filters = dict(item.split('=') for item in filters_str.split(','))
    except ValueError:
        print(f"formato gruppo sbagliato: '{group_def}'. Usa 'Nome|col=val,col=val'")
        exit()
    
    temp_df = df.copy()
    filtri_validi = True

    #Applicazione filtri
    for col, val in filters.items():
        #Rimuovo spazi bianchi
        col = col.strip()
        val = val.strip()

        if col not in temp_df.columns:
            print(f"  ATTENZIONE: La colonna '{col}' non esiste nel CSV. Filtro ignorato.")
            continue
        
        #Applica filtro al dataframe
        temp_df = temp_df[temp_df[col] == val]
    
    if len(temp_df) == 0:
        print(f"Il gruppo '{label_name}' è vuoto con i filtri: {filters}")
        continue
    
    #Colonne per il plotting
    temp_df["group_name"] = label_name.strip()
    temp_df["plot_label"] = temp_df["artista"] + " -> " + label_name.strip()
    
    print(f"Gruppo '{label_name}': {len(temp_df)} immagini.")
    df_plot_list.append(temp_df)

if not df_plot_list:
    print("Nessun dato da plottare dopo il filtraggio.")
    exit()

df_plot = pd.concat(df_plot_list, ignore_index=True)

#--- CARICAMENTO MODELLO ---
try:
    print(f"\nCaricamento modello da: {args.modello}")
    modello_base = load_model(args.modello)
    nome_modello = modello_base.name.lower().replace("-", "").replace("_", "")
    preprocess_input = mappa_funzioni[nome_modello]
    layer_stile = estraiUltimoLayer(modello_base)
    print("Modello caricato e layer estratto.")
except Exception as e:
    print(f"Errore caricamento modello: {e}")
    exit()

#--- ESTRAZIONE FEATURE ---
feature_list = []
labels_list = []
meta_list = []

print(f"\nEstrazione feature ({len(df_plot)} immagini)...")

for i, row in df_plot.iterrows():
    try:
        img = load_img(row["path"], target_size=input_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        pred = layer_stile.predict(x).flatten()
        
        feature_list.append(pred)
        # Salviamo la label univoca (es: "Durer -> Originali")
        labels_list.append(row["plot_label"])
        # Salviamo i metadati grezzi per gestire colori/forme
        meta_list.append({"artista": row["artista"], "gruppo": row["group_name"]})

    except Exception as e:
        print(f"  Errore su {row['filename']}: {e}")

if len(feature_list) == 0:
    print("ERRORE: Nessuna feature estratta.")
    exit()

X = np.array(feature_list)

#--- CALCOLO T-SNE ---
# Perplexity dinamica
if len(df_plot) < 100:
    p = 10
elif 100 <= len(df_plot) <= 500:
    p = 30
elif 501 <= len(df_plot) <= 1000:
    p = 50
else:
    p = 100

print(f"Calcolo t-SNE su {len(df_plot)} punti (Perplexity={p})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=p)
Y = tsne.fit_transform(X)

#--- PLOT DEI RISULTATI ---
plt.figure(figsize=(16, 10))

#Liste uniche per mappature
unique_artists = sorted(list(set(m["artista"] for m in meta_list)))
unique_groups = sorted(list(set(m["gruppo"] for m in meta_list)))
unique_labels = sorted(list(set(labels_list)))

#Colori per Artista
cmap = cm.get_cmap('gist_rainbow', len(unique_artists))
artist_color_map = {artist: cmap(i) for i, artist in enumerate(unique_artists)}

#Forme per Gruppo
markers_avail = ['o', '^', 's', 'P', '*', 'X', 'D', 'v', '<', '>']
group_marker_map = {grp: markers_avail[i % len(markers_avail)] for i, grp in enumerate(unique_groups)}

#Plot dei dati
for label in unique_labels:
    
    indices = [i for i, x in enumerate(labels_list) if x == label]
    
    if not indices:
        continue
    
    # Recuperiamo i metadati dal primo elemento trovato (sono uguali per tutti gli indici della label)
    first_idx = indices[0]
    artist = meta_list[first_idx]["artista"]
    grp = meta_list[first_idx]["gruppo"]
    
    colore = artist_color_map.get(artist, "black")
    marker = group_marker_map.get(grp, "o")
    
    plt.scatter(
        Y[indices, 0], 
        Y[indices, 1], 
        c=[colore], 
        marker=marker, 
        label=label, 
        s=80,
        alpha=0.75,
        edgecolors='w',
        linewidth=0.5
    )

plt.title(f"t-SNE Analysis: {', '.join(unique_groups)}", fontsize=16, fontweight='bold')
plt.legend(
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    borderaxespad=0., 
    fontsize=10,
    title="Artista -> Gruppo"
)
plt.tight_layout()

#--- SALVATAGGIO ---
cartella = "t-sne_plots/dynamic_plots/"
os.makedirs(cartella, exist_ok=True)

grp_names = "-".join(unique_groups).replace(" ", "_").replace("|", "")
grp_names = grp_names[:50]
nome_file = f"tsne_{grp_names}.png"

percorso_completo = os.path.join(cartella, nome_file)
plt.savefig(percorso_completo, bbox_inches='tight', dpi=300)

print(f"Plot salvato con successo in: {percorso_completo}")