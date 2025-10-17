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

# Define the dataset's root folder name and the training path
DATASET_ROOT_FOLDER = 'brain-tumor-mri-dataset'
TRAIN_DATA_PATH = os.path.join(DATASET_ROOT_FOLDER, 'Training')

# Function to download and extract dataset if not already present
@st.cache_resource
def download_kaggle_dataset():
    api = KaggleApi()
    api.authenticate()
    
    if not os.path.exists(TRAIN_DATA_PATH):
        st.info("Dataset not found. Downloading and extracting...")
        try:
            api.dataset_download_files('masoudnickparvar/brain-tumor-mri-dataset', path='.', unzip=True)
            if os.path.exists(TRAIN_DATA_PATH):
                 st.success("Dataset downloaded and extracted successfully.")
            else:
                 st.warning(f"Downloaded root folder found, but training path '{TRAIN_DATA_PATH}' is missing. Check dataset structure.")
        except Exception as e:
            st.error(f"Kaggle Download Failed: {e}")
    else:
        st.write("Dataset already exists.")

download_kaggle_dataset()

# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------
img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

if not os.path.exists(TRAIN_DATA_PATH):
    st.error(f"Required data folder '{TRAIN_DATA_PATH}' not found. Cannot proceed.")
    st.stop()
else:
    train_ds = train_gen.flow_from_directory(
        TRAIN_DATA_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_ds = train_gen.flow_from_directory(
        TRAIN_DATA_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    num_classes = train_ds.num_classes
    class_labels = list(train_ds.class_indices.keys())
    st.write(f"Found {num_classes} classes: {class_labels}")

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    st.title("Brain Tumor Classification (4 Classes)")
# ----------------------------------------------------------------------
# STREAMLIT UI AND EXECUTION
# ----------------------------------------------------------------------
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            history = model.fit(train_ds, validation_data=val_ds, epochs=10) 
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
        
        pred_array = model.predict(x)[0]
        
        pred_class_index = np.argmax(pred_array)
        
        predicted_label = class_labels[pred_class_index]
        confidence = pred_array[pred_class_index]

        st.write(f"Prediction: **{predicted_label}** (Confidence: {confidence:.2f})")