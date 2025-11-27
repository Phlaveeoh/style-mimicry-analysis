import os
import pandas as pd
import numpy as np
from pathlib import Path
from indicizzaDataset import indicizzaDataset

from keras.applications.convnext import ConvNeXtBase, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


# Caricamento del dataset indicizzato
percorsoDataset = "C:\\Users\\Flavio\\Desktop\\dataset"
df = indicizzaDataset(percorsoDataset)

# Preparazione del dataset per il fine tuning
artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
    ]
artista_target = artisti[0]
print("fine tuning su albrecht-durer")

df_pos = df[(df["stato"] == "originale") & (df["artista"] == artista_target)].copy()
df_neg = df[(df["stato"] == "originale") & (df["artista"] != artista_target)].copy()

df_pos["label"] = "Albrecht Durer"
df_neg["label"] = "Altro Artista"

df_train = pd.concat([df_pos, df_neg]).reset_index(drop=True)

# Creazione generatori da DataFrame
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    df_train,
    x_col="path",
    y_col="label",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=16,
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    df_train,
    x_col="path",
    y_col="label",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=16,
    subset="validation",
    shuffle=False
)

# Costruzione modello
base = ConvNeXtBase(weights="imagenet", include_top=False)

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
out = Dense(2, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

# Addestrare solo i top layers
for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# Fine tuning della parte alta della rete
for layer in base.layers[:200]:
    layer.trainable = False
for layer in base.layers[200:]:
    layer.trainable = True

opt = SGD(learning_rate=1e-4, momentum=0.9)

model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

model.save(f"convnext_finetuned_{artista_target}.keras")