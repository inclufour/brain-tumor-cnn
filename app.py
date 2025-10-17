import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

# Set Kaggle credentials securely from Streamlit secrets
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

def download_kaggle_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset_folder = 'brain-tumor-mri-dataset'
    if not os.path.exists(dataset_folder):
        api.dataset_download_files('masoudnickparvar/brain-tumor-mri-dataset', path='.', unzip=True)
        st.write("Dataset downloaded and extracted.")
    else:
        st.write("Dataset already exists.")

download_kaggle_dataset()

img_size = (150, 150)
batch_size = 16

# Data generators with validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training dataset generator
train_ds = datagen.flow_from_directory(
    'brain-tumor-mri-dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation dataset generator
val_ds = datagen.flow_from_directory(
    'brain-tumor-mri-dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

st.title("Brain Tumor Detection - MRI Classification")

if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        history = model.fit(train_ds, validation_data=val_ds, epochs=5)
        st.success("Training complete!")
        st.line_chart({
            "Training Accuracy": history.history['accuracy'],
            "Validation Accuracy": history.history['val_accuracy']
        })

uploaded_file = st.file_uploader("Upload an MRI image for tumor detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=img_size)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    label = "Tumor Detected" if pred > 0.5 else "No Tumor Detected"
    st.write(f"Prediction: {label} (Confidence: {pred:.2f})")
