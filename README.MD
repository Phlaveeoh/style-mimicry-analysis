# Robust Style Analysis

Questo repository raccoglie script per l’analisi delle feature map di immagini protette e non protette attraverso reti convoluzionali pre-addestrate, con visualizzazione tramite t-SNE.

## Obiettivo

Il progetto nasce come approfondimento dello studio presentato in [“Robust Style Mimicry”](https://arxiv.org/abs/2406.12027), con l’obiettivo di:

- Analizzare le feature map estratte da immagini di diversi artisti, confrontando versioni protette (con filtri di protezione come “antidb”) e non protette.
- Studiare la vicinanza delle rappresentazioni nello spazio delle feature latenti in diversi layer della rete convoluzionale.
- Visualizzare i risultati tramite t-SNE per evidenziare come la protezione influisce sulla rappresentazione interna delle immagini.