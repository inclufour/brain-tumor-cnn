# Importing Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
import matplotlib.pyplot as plt
import numpy as np

# Loading the Dataset
# Run these commands in your environment to download the dataset if needed:
# !kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
# !unzip brain-mri-images-for-brain-tumor-detection.zip -d data/

# Preparing the Data
img_size = (150, 150)
batch_size = 16
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Loading the Images from Folders
train_ds = train_gen.flow_from_directory(
    'data/brain_tumor_dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_ds = train_gen.flow_from_directory(
    'data/brain_tumor_dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Building the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling
