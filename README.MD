# Robust Style Analysis

Questo repository raccoglie script per l’analisi delle feature map di immagini protette e non protette ottenute attraverso reti convoluzionali pre-addestrate, con visualizzazione tramite t-SNE.

## Obiettivo

Il progetto nasce come approfondimento dello studio presentato in [“Robust Style Mimicry”](https://arxiv.org/abs/2406.12027), con l’obiettivo di valutare l’efficacia delle attuali tecniche di protezione delle immagini digitali. In particolare, si mira a dimostrare che le protezioni esistenti non garantiscono una sufficiente separazione nello spazio delle feature latenti tra immagini protette e corrispettive immagini originali, consentendo a un agente malintenzionato di replicare lo stile dell’immagine originale sfruttando le feature apprese dalle immagini nonostante la protezione.

## Quanto fatto finora

- Analizzare le feature map estratte da immagini di diversi artisti, confrontando versioni protette (con filtri di protezione come “antidb”) e non protette.
- Studiare la vicinanza delle rappresentazioni nello spazio delle feature latenti in diversi layer della rete convoluzionale.
- Visualizzare i risultati tramite t-SNE per evidenziare come la protezione influisce sulla rappresentazione interna delle immagini.
- Eseguito il fine tuning su un modello (**CONVNEXT**) per migliorare il riconoscimento dello stile dei 5 artisti col fine di massimizzare la distanza tra le feature map.
- Fatti alcuni plot tra immagini originali e generate usando come modello quello ottenuto tramite il fine tuning per evidenziare i cluster.