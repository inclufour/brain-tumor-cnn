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

# Define the dataset's root folder name
DATASET_ROOT_FOLDER = 'brain-tumor-mri-dataset'
# Anticipated path *if* extraction is perfect
DEFAULT_TRAIN_PATH = os.path.join(DATASET_ROOT_FOLDER, 'Training')

@st.cache_resource
def download_and_get_data_path():
    """Downloads the dataset and returns the correct path for the ImageDataGenerator."""
    api = KaggleApi()
    api.authenticate()
    
    # 1. Download and Unzip (if the root folder doesn't exist)
    if not os.path.exists(DATASET_ROOT_FOLDER):
        st.info("Dataset root folder not found. Downloading and extracting...")
        try:
            api.dataset_download_files('masoudnickparvar/brain-tumor-mri-dataset', path='.', unzip=True)
            st.success("Initial download and extraction complete.")
        except Exception as e:
            st.error(f"Kaggle Download Failed: {e}")
            return None # Return None if download fails

    # 2. Check for correct path or nested structure
    
    # Check 1: Is the expected path correct?
    if os.path.exists(DEFAULT_TRAIN_PATH):
        st.write("Dataset structure is correct.")
        return DEFAULT_TRAIN_PATH
    
    # Check 2: Look for a nested structure if the root folder exists
    if os.path.exists(DATASET_ROOT_FOLDER):
        contents = os.listdir(DATASET_ROOT_FOLDER)
        
        # This logic handles a common case where the ZIP extracts to a single nested folder
        if len(contents) == 1 and os.path.isdir(os.path.join(DATASET_ROOT_FOLDER, contents[0])):
            nested_folder = contents[0]
            corrected_path = os.path.join(DATASET_ROOT_FOLDER, nested_folder, 'Training')
            
            if os.path.exists(corrected_path):
                st.warning(f"Dataset structure was nested. Path corrected to: {corrected_path}")
                return corrected_path

    # 3. If all checks fail
    st.error(f"Required data path is missing. Tried: {DEFAULT_TRAIN_PATH}")
    return None

# Execute the download and get the definitive data path
TRAIN_DATA_PATH = download_and_get_data_path()

# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------

if TRAIN_DATA_PATH is None:
    st.stop() # Stop execution if the path could not be determined

img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Now, we use the guaranteed-correct TRAIN_DATA_PATH
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

# ----------------------------------------------------------------------
# MODEL DEFINITION (no change)
# ----------------------------------------------------------------------
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
# STREAMLIT UI AND EXECUTION (no change)
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