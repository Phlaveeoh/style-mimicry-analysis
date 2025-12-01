import pandas as pd
from keras.applications.convnext import ConvNeXtBase, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from indicizzaDataset import indicizza_dataset_completo

# Caricamento del dataset indicizzato
percorsoDataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"
df = indicizza_dataset_completo(percorsoDataset)

# Preparazione del dataset per il fine tuning
artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
    ]

artista_target = artisti[1]
print("fine tuning su:", artista_target)

# 1. BILANCIAMENTO CLASSI
df_pos = df[
    (df["artista"] == artista_target) & 
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy()
# Prendo dagli altri artisti un numero di immagini UGUALE a quello del target
df_neg_totale = df[
    (df["artista"] != artista_target) & 
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy()

df_neg = df_neg_totale.sample(n=len(df_pos), random_state=42).copy()

df_pos["label"] = "Artista Target"
df_neg["label"] = "Altro Artista"

df_train = pd.concat([df_pos, df_neg]).reset_index(drop=True)
print(f"Dataset bilanciato: {len(df_pos)} positivi vs {len(df_neg)} negativi.")

# 2. DATA AUGMENTATION (Solo Flip e Crop implicito così da non strravolgere lo stile)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='reflect'
)

train_gen = datagen.flow_from_dataframe(
    df_train,
    x_col="path",
    y_col="label",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=8,
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    df_train,
    x_col="path",
    y_col="label",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=8,
    subset="validation",
    shuffle=False
)

# Costruzione del modello
base = ConvNeXtBase(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelo tutti i layer del base model
base.trainable = False 

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
out = Dense(2, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)


model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping per evitare overfitting
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

print("Inizio training")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop]
)

model.save(f"convnext_finetuned_{artista_target}.keras")