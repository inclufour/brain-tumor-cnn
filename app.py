import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
DATASET_ROOT_FOLDER = 'brain-tumor-mri-dataset'
FINAL_TRAIN_PATH = os.path.join(DATASET_ROOT_FOLDER, 'Training')

@st.cache_resource
def download_and_clean_data():
    """
    Downloads the dataset and uses a shell command to correct the folder structure.
    Returns the FINAL_TRAIN_PATH if successful, or None otherwise.
    """
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(DATASET_ROOT_FOLDER):
        st.info("Dataset not found. Downloading and extracting...")
        try:
            api.dataset_download_files('masoudnickparvar/brain-tumor-mri-dataset', path='.', unzip=True)
            st.success("Initial download and extraction complete.")
        except Exception as e:
            st.error(f"Kaggle Download Failed: {e}")
            return None

    # Check for the common nested folder issue and use shell commands to fix it.
    if not os.path.exists(FINAL_TRAIN_PATH):
        st.warning("Training path missing. Checking for nested folder...")
        
        # Get the list of contents in the root folder
        contents = os.listdir(DATASET_ROOT_FOLDER)
        
        # We look for a single nested folder (e.g., 'brain_tumor_mri_dataset')
        if len(contents) == 1 and os.path.isdir(os.path.join(DATASET_ROOT_FOLDER, contents[0])):
            nested_dir = contents[0]
            NESTED_TRAIN_PATH = os.path.join(DATASET_ROOT_FOLDER, nested_dir, 'Training')
            
            if os.path.exists(NESTED_TRAIN_PATH):
                st.info(f"Nested structure found: {nested_dir}. Fixing with shell commands...")

                try:
                    # Shell command 1: Move all contents from the nested directory up to the root
                    source = os.path.join(DATASET_ROOT_FOLDER, nested_dir, '*')
                    destination = DATASET_ROOT_FOLDER
                    # Using 'mv' command to move contents up one level
                    os.system(f'mv {source} {destination}')

                    # Shell command 2: Remove the empty nested directory
                    os.system(f'rmdir {os.path.join(DATASET_ROOT_FOLDER, nested_dir)}')
                    
                    st.success("File structure corrected successfully using shell commands.")
                    # After the shell move, FINAL_TRAIN_PATH should now exist.
                    if os.path.exists(FINAL_TRAIN_PATH):
                        return FINAL_TRAIN_PATH
                    else:
                        st.error("Correction failed: 'Training' folder still not found after shell move.")
                        return None
                except Exception as e:
                    st.error(f"Shell command failed: {e}")
                    return None

    if os.path.exists(FINAL_TRAIN_PATH):
        return FINAL_TRAIN_PATH
    
    st.error(f"Required data path is missing: {FINAL_TRAIN_PATH}. Cannot proceed.")
    return None

# Execute the download and get the definitive data path
TRAIN_DATA_PATH = download_and_clean_data()

# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------

if TRAIN_DATA_PATH is None:
    st.stop()

img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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
# MODEL DEFINITION
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