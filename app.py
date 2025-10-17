import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup Kaggle credentials securely from Streamlit secrets
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# Function to download and extract dataset if not already present
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
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Check if dataset folder exists before loading
dataset_path = 'brain-tumor-mri-dataset'
if not os.path.exists(dataset_path):
    st.error(f"Dataset folder '{dataset_path}' not found. Please check download step.")
else:
    train_ds = train_gen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_ds = train_gen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    st.title("Brain Tumor Detection")

    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            history = model.fit(train_ds, validation_data=val_ds, epochs=5)
            st.success("Training complete!")
            st.line_chart({
                "Training Accuracy": history.history['accuracy'],
                "Validation Accuracy": history.history['val_accuracy']
            })

    uploaded_file = st.file_uploader("Upload MRI Image for Prediction", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = load_img(uploaded_file, target_size=img_size)
        st.image(img, use_column_width=True, caption="Uploaded image")

        x = img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)[0][0]
        label = "Tumor Detected" if pred > 0.5 else "No Tumor Detected"
        st.write(f"Prediction: {label} (Confidence: {pred:.2f})")
