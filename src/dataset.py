import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, patient_ids, transform=None):
        self.df = pd.read_csv(csv_path)

        # Convert both sides to string for safe matching
        self.df["Patient ID"] = self.df["Patient ID"].astype(str)
        patient_ids = [str(pid).strip() for pid in patient_ids]

        # Filter
        self.df = self.df[self.df["Patient ID"].isin(patient_ids)]
        
        # Remove rows where image file does not exist
        self.df = self.df[
            self.df["Image Index"].apply(
                lambda x: os.path.exists(os.path.join(image_dir, x))
            )
        ]

        print("Dataset loaded. Rows after filtering:", len(self.df))

        self.image_dir = image_dir
        self.transform = transform

        self.label_map = {
            "Atelectasis": 0,
            "Cardiomegaly": 1,
            "Effusion": 2,
            "Infiltration": 3,
            "Mass": 4,
            "Nodule": 5,
            "Pneumonia": 6,
            "Pneumothorax": 7,
            "Consolidation": 8,
            "Edema": 9,
            "Emphysema": 10,
            "Fibrosis": 11,
            "Pleural_Thickening": 12,
            "Hernia": 13
        }

    def __len__(self):
        return len(self.df)

    def encode_labels(self, label_str):
        labels = torch.zeros(14)
        if label_str != "No Finding":
            for l in label_str.split("|"):
                if l in self.label_map:
                    labels[self.label_map[l]] = 1
        return labels

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])

        image = Image.open(img_path).convert("RGB")
        labels = self.encode_labels(row["Finding Labels"])

        if self.transform:
            image = self.transform(image)

        return image, labels