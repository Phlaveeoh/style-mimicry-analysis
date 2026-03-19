import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Import modelli
from keras.applications.convnext import ConvNeXtBase, preprocess_input as convnext_pre

# Import utils personali
from utils.estraiLayer import estraiUltimoLayer 
from utils.indicizzaDataset import indicizza_dataset_completo

# --- CONFIGURAZIONE ARGPARSE ---
parser = argparse.ArgumentParser(
    description="Crea un plot t-SNE dinamico confrontando gruppi di immagini."
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
    help="Definisci un gruppo da plottare. Sintassi: 'Label|colonna=valore,colonna2=valore'. "
         "Colonne valide: categoria, tipo, protezione, metodo_processamento. "
         "Esempio: 'Originali|tipo=Original,categoria=Training'"
)

args = parser.parse_args()

# --- CONFIGURAZIONI INIZIALI ---
mappa_funzioni = {
    "convnextbase": convnext_pre,
    "functional4": convnext_pre # Mappatura per modelli decapitati salvati custom
}

input_size = (224, 224)

artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
]

# --- CARICAMENTO E FILTRAGGIO DATASET ---
print(f"Indicizzazione dataset da: {args.dataset}")
df = indicizza_dataset_completo(args.dataset)

if df.empty:
    print("Dataset vuoto o non trovato.")
    exit()

# Filtro preliminare sugli artisti
df = df[df["artista"].isin(artisti)].copy()

df_plot_list = []
print(f"Generazione gruppi basata su {len(args.group)} definizioni...")

# Parsing dei gruppi dinamici
for group_def in args.group:
    try:
        # Divide nome gruppo dai filtri
        label_name, filters_str = group_def.split('|')
        filters = dict(item.split('=') for item in filters_str.split(','))
    except ValueError:
        print(f"Formato gruppo errato: '{group_def}'. Usa 'Nome|col=val,col=val'")
        exit()
    
    temp_df = df.copy()

    # Applicazione filtri
    for col, val in filters.items():
        col = col.strip()
        val = val.strip()

        if col not in temp_df.columns:
            print(f"La colonna '{col}' non esiste nel dataset. Filtro ignorato.")
            continue
        
        # Applica filtro al dataframe
        temp_df = temp_df[temp_df[col] == val]
    
    if len(temp_df) == 0:
        print(f"ATTENZIONE: Il gruppo '{label_name}' è vuoto con i filtri {filters}. Ignorato.")
        continue
    
    # Colonne per il plotting
    temp_df["group_name"] = label_name.strip()
    # Pulisce il prefisso "wikiart_" per una visualizzazione più chiara nel plot
    nome_artista_pulito = temp_df["artista"].str.replace("wikiart_", "").str.replace("-", " ").str.title()
    temp_df["plot_label"] = nome_artista_pulito + " -> " + label_name.strip()
    
    print(f"Gruppo '{label_name}': {len(temp_df)} immagini isolate.")
    df_plot_list.append(temp_df)

if not df_plot_list:
    print("ERRORE: Nessun dato da plottare dopo l'applicazione dei filtri.")
    exit()

df_plot = pd.concat(df_plot_list, ignore_index=True)

# --- CARICAMENTO MODELLO ---
try:
    print(f"\nCaricamento modello da: {args.modello}")
    modello_base = load_model(args.modello)
    nome_modello = modello_base.name.lower().replace("-", "").replace("_", "")
    
    # Selezione dinamica della funzione di preprocessing
    preprocess_input = mappa_funzioni.get(nome_modello, convnext_pre) # Default di fallback
    
    layer_stile = estraiUltimoLayer(modello_base)
    print(f"Modello caricato. Layer estrattivo: {layer_stile.output_shape}")
except Exception as e:
    print(f"ERRORE caricamento modello: {e}")
    exit()

# --- ESTRAZIONE FEATURE ---
feature_list = []
labels_list = []
meta_list = []

print(f"\nAvvio estrazione feature vettoriali ({len(df_plot)} campioni)...")

for i, row in df_plot.iterrows():
    try:
        img = load_img(row["path"], target_size=input_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        pred = layer_stile.predict(x, verbose=0).flatten()
        
        feature_list.append(pred)
        labels_list.append(row["plot_label"])
        
        artista_display = row["artista"].replace("wikiart_", "").replace("-", " ").title()
        meta_list.append({"artista": artista_display, "gruppo": row["group_name"]})

    except Exception as e:
        print(f"Errore inferenza su {row['filename']}: {e}")

if len(feature_list) == 0:
    print("ERRORE FATALE: Nessun vettore estratto.")
    exit()

X = np.array(feature_list)

p = 20

print(f"Inizializzazione t-SNE su {len(df_plot)} punti (Perplexity ottimizzata a {p})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=p)
Y = tsne.fit_transform(X)

# --- PLOT DEI RISULTATI ---
plt.figure(figsize=(16, 10))

unique_artists = sorted(list(set(m["artista"] for m in meta_list)))
unique_groups = sorted(list(set(m["gruppo"] for m in meta_list)))
unique_labels = sorted(list(set(labels_list)))

# Assegnazione Colore = Artista
cmap = cm.get_cmap('gist_rainbow', len(unique_artists))
artist_color_map = {artist: cmap(i) for i, artist in enumerate(unique_artists)}

# Assegnazione Forma = Gruppo/Filtro
markers_avail = ['o', '^', 's', 'P', '*', 'X', 'D', 'v', '<', '>']
group_marker_map = {grp: markers_avail[i % len(markers_avail)] for i, grp in enumerate(unique_groups)}

for label in unique_labels:
    indices = [i for i, x in enumerate(labels_list) if x == label]
    
    if not indices:
        continue
    
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
        s=120, # Aumentata dimensione marker per visibilità
        alpha=0.8,
        edgecolors='w',
        linewidth=0.8
    )

plt.title(f"Gruppi: {', '.join(unique_groups)}", fontsize=12, fontweight='bold')
plt.legend(
    bbox_to_anchor=(1.02, 1), 
    loc='upper left', 
    borderaxespad=0., 
    fontsize=11,
    title="Classe Stilistica"
)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# --- SALVATAGGIO ---
def estrai_valore_dominante(colonna, fallback):
    if colonna in df_plot.columns:
        valori = [str(v) for v in df_plot[colonna].unique() if str(v).lower() not in ['none', 'nan', '']]
        if valori:
            return "_".join(sorted(valori))
    return fallback

dominio = estrai_valore_dominante("categoria", "Sconosciuto")
protezione = estrai_valore_dominante("protezione", "Nessuna_Protezione")
purificazione = estrai_valore_dominante("metodo_processamento", "Nessuna_Purificazione")

cartella = os.path.join("t-sne_plots", dominio, protezione, purificazione)
os.makedirs(cartella, exist_ok=True)

nome_file = f"tsne_plot.png"
percorso_completo = os.path.join(cartella, nome_file)

plt.savefig(percorso_completo, bbox_inches='tight', dpi=300)
print(f"Generazione t-SNE completata. File scritto in: {percorso_completo}")