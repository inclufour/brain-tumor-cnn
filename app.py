import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import zipfile # <-- NEW IMPORT
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuration ---
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
DATASET_NAME = 'masoudnickparvar/brain-tumor-mri-dataset'
DATASET_ROOT_FOLDER = 'brain-tumor-mri-dataset'
FINAL_TRAIN_PATH = os.path.join(DATASET_ROOT_FOLDER, 'Training')

@st.cache_resource
def download_and_unzip_data():
    """
    Downloads the dataset as a ZIP, then manually unzips it to ensure correct path structure.
    Returns the FINAL_TRAIN_PATH if successful, or None otherwise.
    """
    api = KaggleApi()
    api.authenticate()
    
    ZIP_FILE_NAME = f"{DATASET_NAME.split('/')[-1]}.zip"
    
    # 1. DOWNLOAD THE ZIP FILE (force unzip=False initially)
    if not os.path.exists(ZIP_FILE_NAME):
        st.info(f"Downloading dataset as {ZIP_FILE_NAME}...")
        try:
            # Download the zip archive to the current directory
            api.dataset_download_files(DATASET_NAME, path='.', unzip=False)
            st.success("Download complete.")
        except Exception as e:
            st.error(f"Kaggle Download Failed: {e}")
            return None

    # 2. MANUAL EXTRACTION (Create the expected root folder and extract)
    
    # Ensure the root folder exists for extraction
    if not os.path.exists(DATASET_ROOT_FOLDER):
        os.makedirs(DATASET_ROOT_FOLDER, exist_ok=True)
        
    # Check if extraction is already complete (by checking the final path)
    if os.path.exists(FINAL_TRAIN_PATH):
        st.write("Dataset already extracted and verified.")
        return FINAL_TRAIN_PATH

    st.info(f"Extracting {ZIP_FILE_NAME} to {DATASET_ROOT_FOLDER}...")
    try:
        with zipfile.ZipFile(ZIP_FILE_NAME, 'r') as zip_ref:
            # zip_ref.extractall(DATASET_ROOT_FOLDER)
            
            # CRITICAL: Find the actual 'Training' folder inside the zip 
            # and extract ONLY its contents to the target path
            
            # The zip contains: brain_tumor_mri_dataset/Training/...
            # We want to extract it to: brain-tumor-mri-dataset/Training/...
            
            zip_contents = zip_ref.namelist()
            # Find the root folder inside the zip, typically the dataset name
            internal_root = next((name for name in zip_contents if name.endswith('/')), None)
            
            if internal_root:
                 # Extracting only the necessary contents
                 for member in zip_contents:
                     if member.startswith(internal_root):
                         # Create new path without the internal_root layer
                         target_path = os.path.join(DATASET_ROOT_FOLDER, member[len(internal_root):])
                         if target_path:
                             zip_ref.extract(member, path=DATASET_ROOT_FOLDER)
                             
                 # Clean up the original zip file to save space (optional)
                 os.remove(ZIP_FILE_NAME)
                 
                 # The extraction is complete, but the files are likely nested one level deeper 
                 # (e.g., brain-tumor-mri-dataset/brain_tumor_mri_dataset/Training)
                 
                 # Since we cannot rely on shutil or os.system, we'll return the nested path
                 # and adjust the flow_from_directory call below.
                 
                 NESTED_PATH = os.path.join(DATASET_ROOT_FOLDER, internal_root.strip('/'), 'Training')
                 if os.path.exists(NESTED_PATH):
                     st.warning(f"Extracted to nested path. Returning: {NESTED_PATH}")
                     return NESTED_PATH
                     
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        # Clean up the zip if extraction failed
        if os.path.exists(ZIP_FILE_NAME):
            os.remove(ZIP_FILE_NAME)
        return None

    st.error("Extraction failed: 'Training' folder not found after extraction.")
    return None

# Execute the download and get the definitive data path
# TRAIN_DATA_PATH will now be the potentially nested, but verified, path
TRAIN_DATA_PATH = download_and_unzip_data()

# ----------------------------------------------------------------------
# DATA LOADING SETUP
# ----------------------------------------------------------------------

if TRAIN_DATA_PATH is None:
    st.stop()

# ... (The rest of the code remains the same, as the TRAIN_DATA_PATH variable is correctly set)

img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# This relies on TRAIN_DATA_PATH holding the *correct* (potentially nested) folder path
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