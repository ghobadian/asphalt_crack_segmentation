"""
Configuration settings for the crack segmentation project.
"""

import os

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'clean')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# === Model Hyperparameters ===
IMG_SIZE = 512
BATCH_SIZE = 14
EPOCHS = 50
LEARNING_RATE = 1e-4

# === Data Split Ratios ===
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 37

# === Create directories if they don't exist ===
for directory in [DATA_CLEAN_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)