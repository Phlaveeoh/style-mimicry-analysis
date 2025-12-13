import pandas as pd
import tensorflow as tf
from keras.applications.convnext import ConvNeXtBase, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from indicizzaDataset import indicizza_dataset_completo

percorsoDataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"
df = indicizza_dataset_completo(percorsoDataset)

#Preparazione dataset
artisti = [
    "wikiart_albrecht-durer",
    "wikiart_alphonse-mucha",
    "wikiart_anna-ostroumova-lebedeva",
    "wikiart_edvard-munch",
    "wikiart_edward-hopper",
]

df_train = df[
    (df["artista"].isin(artisti)) & 
    (df["categoria"] == "Training") &
    (df["tipo"] == "Original")
].copy().reset_index(drop=True)


train_df, val_df = train_test_split(
    df_train, 
    test_size=0.2, 
    stratify=df_train["artista"],
    random_state=42
)

#Data Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )


BATCH_SIZE = 16
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="artista",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="path",
    y_col="artista",
    class_mode="categorical",
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Costruzione Modello")

base = ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
out = Dense(5, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)

#Allenamento iniziale della testa
print("Primo training della testa della CNN")

loss_function = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=loss_function,
    metrics=["accuracy"]
)

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

# Fine tuning totale
print("Fine Tuning del modello completo")

#Scongelo solo gli ultimi 20 layer del modello base
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-6), 
    loss=loss_function,
    metrics=["accuracy"]
)

early_stop_fine = EarlyStopping(
    monitor='val_loss', 
    patience=5,
    min_delta=0.05,
    restore_best_weights=True
)

history_phase2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    callbacks=[early_stop_fine]
)

#Salvo il modello dopo il fine tuning
print("Salvataggio del modello fine-tuned")
nome_file = f"{model.name}finetuned.keras"
model.save(nome_file)