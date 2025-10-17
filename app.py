import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import zipfile # Standard library - no requirements.txt change needed
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
DATASET_NAME = 'masoudnickparvar/brain-tumor-mri-dataset'
DATASET_ROOT_FOLDER = 'brain-tumor-mri-dataset'
FINAL_TRAIN_PATH = os.path.join(DATASET_ROOT_FOLDER, 'Training')

@st.cache_resource
def download_and_extract_data():
    """
    Downloads the dataset as a ZIP, then manually extracts it by stripping 
    the internal top-level directory to ensure the correct path structure.
    Returns the FINAL_TRAIN_PATH if successful, or None otherwise.
    """
    api = KaggleApi()
    api.authenticate()
    
    ZIP_FILE_NAME = f"{DATASET_NAME.split('/')[-1]}.zip"
    
    # 1. DOWNLOAD THE ZIP FILE
    if not os.path.exists(ZIP_FILE_NAME):
        st.info(f"Downloading dataset as {ZIP_FILE_NAME}...")
        try:
            # Force unzip=False to guarantee we have the zip file
            api.dataset_download_files(DATASET_NAME, path='.', unzip=False)
            st.success("Download complete.")
        except Exception as e:
            st.error(f"Kaggle Download Failed: {e}")
            return None

    # 2. MANUAL EXTRACTION AND PATH STRIPPING
    
    # Check if extraction is already complete (by checking the final path)
    if os.path.exists(FINAL_TRAIN_PATH):
        st.write("Dataset already extracted and verified.")
        return FINAL_TRAIN_PATH

    st.info(f"Extracting {ZIP_FILE_NAME} and correcting paths...")
    
    try:
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            zip_contents = zip_ref.namelist()
            
            # Identify the single, unwanted top-level folder inside the zip 
            # (e.g., 'brain_tumor_mri_dataset/')
            internal_root = zip_contents[0].split('/')[0] + '/' 
            
            for member in zip_contents:
                if member.startswith(internal_root) and member != internal_root:
                    
                    # Strip the leading folder name (e.g., 'brain_tumor_mri_dataset/')
                    relative_path = member[len(internal_root):]
                    
                    # Define the final target path
                    target_path = os.path.join(DATASET_ROOT_FOLDER, relative_path)
                    
                    # If it's a directory, create it
                    if member.endswith('/'):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        # Ensure the target directory exists before extracting the file
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        # Extract the file directly to the correct location
                        # This avoids the nested folder
                        with open(target_path, 'wb') as outfile:
                            outfile.write(zip_ref.read(member))

        # Cleanup the original zip file
        os.remove(ZIP_FILE_NAME)
        
        # FINAL CHECK: The required 'Training' folder MUST exist now
        if os.path.exists(FINAL_TRAIN_PATH):
            st.success("Extraction and path correction successful.")
            return FINAL_TRAIN_PATH
            
    except Exception as e:
        st.error(f"Critical Extraction Failure: {e}")
        return None

    st.error("Extraction failed: 'Training' folder not found after extraction. Manual path stripping was unsuccessful.")
    return None

# Execute the download and get the definitive data path
TRAIN_DATA_PATH = download_and_extract_data()

# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------

if TRAIN_DATA_PATH is None:
    st.stop()

img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Use the guaranteed-correct TRAIN_DATA_PATH
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