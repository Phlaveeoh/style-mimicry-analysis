import os
import random

def prendiImmagini(cartella):
    immagini = []
    for file in os.listdir(cartella):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            immagini.append(os.path.join(cartella, file))
    return immagini