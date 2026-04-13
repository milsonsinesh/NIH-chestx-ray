# src/config.py

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 14
EPOCHS = 10
LR = 1e-4

DEVICE = "cuda"

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]
