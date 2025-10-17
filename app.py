import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi 

# --- CONFIGURATION: FINAL CORRECT SLUG ---
# Slug is correct: inclufour/final-brain-tumor
DEFAULT_DATASET_SLUG = 'inclufour/final-brain-tumor' 
# ------------------------------------------

# --- NEW: ROOT FOLDER NAME CORRECTED ---
# This matches the directory created when the ZIP file is extracted (as per your image).
DATASET_ROOT_FOLDER = 'tumordataset' 
# ---------------------------------------

# --- UPDATED SUBDIRECTORY NAMES (LOWERCASE) ---
# Directories inside 'tumordataset' are 'train' and 'val' (lowercase).
TRAIN_DIR = os.path.join(DATASET_ROOT_FOLDER, 'train')
VAL_DIR = os.path.join(DATASET_ROOT_FOLDER, 'val')
# ----------------------------------------------

# Determine the dataset slug to use (prefers secret, falls back to default)
DATASET_SLUG = st.secrets.get("DATASET_SLUG", DEFAULT_DATASET_SLUG)

@st.cache_resource
def download_data(dataset_slug):
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        st.write("Dataset already exists locally.")
        return True

    if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
        st.error("Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY) not found in Streamlit Secrets. Cannot download private dataset.")
        st.stop()
    
    try:
        os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

        api = KaggleApi()
        api.authenticate()
        
        st.info(f"Downloading and extracting private dataset: {dataset_slug}...")
        api.dataset_download_files(dataset_slug, path='.', unzip=True)
        st.success("Download and extraction complete.")
        
        # Check if the expected directories exist after extraction 
        if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
            return True
        else:
            # This error will now display the correct path: 'tumordataset/train'
            st.error(f"Downloaded root folder found, but training path '{TRAIN_DIR}' is missing. Check dataset structure inside the '{DATASET_ROOT_FOLDER}' folder.")
            return False

    except Exception as e:
        st.error(f"Kaggle Download Failed. Ensure the slug '{dataset_slug}' is correct and Kaggle secrets are valid. Error: {e}")
        return False

# Execute download with the determined slug
if not download_data(DATASET_SLUG):
    st.error(f"Required data folders '{TRAIN_DIR}' and '{VAL_DIR}' not found. Cannot proceed.")
    st.stop()


# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------
img_size = (150, 150)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10) 
val_gen = ImageDataGenerator(rescale=1./255) 

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary', 
    shuffle=True
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary', 
    shuffle=False
)

num_classes = train_ds.num_classes 
class_labels = list(train_ds.class_indices.keys()) 
st.write(f"Found {num_classes} classes: {class_labels}")


# ----------------------------------------------------------------------
# MODEL DEFINITION
# ----------------------------------------------------------------------
if 'model' not in st.session_state:
    st.session_state.model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(1, activation='sigmoid')
    ])
    
    st.session_state.model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    st.session_state.class_labels = class_labels

# ----------------------------------------------------------------------
# TRAINING BUTTON & LOGIC (Template)
# ----------------------------------------------------------------------
st.sidebar.header("Model Training")
epochs = st.sidebar.number_input("Epochs", min_value=1, value=5, step=1)

if st.sidebar.button("Start Training"):
    with st.spinner(f"Training model for {epochs} epochs..."):
        history = st.session_state.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds
        )
    st.success("Training complete! Run the prediction below.")
    st.subheader("Training History")
    st.line_chart(history.history['accuracy'], label='Training Accuracy')
    st.line_chart(history.history['val_accuracy'], label='Validation Accuracy')

# ----------------------------------------------------------------------
# USER PREDICTION LOGIC
# ----------------------------------------------------------------------
st.header("Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=img_size)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    
    pred_prob = st.session_state.model.predict(x)[0][0] 
    
    threshold = 0.5
    if pred_prob >= threshold:
        predicted_label = st.session_state.class_labels[1] 
        confidence = pred_prob
    else:
        predicted_label = st.session_state.class_labels[0] 
        confidence = 1.0 - pred_prob

    st.header("Prediction Result")
    st.write(f"The model predicts: **{predicted_label.replace('_', ' ').upper()}**")
    st.write(f"Confidence: **{confidence:.2f}**")