import pandas as pd
import os
import numpy as np

df = pd.read_csv("data/raw/Data_Entry_2017.csv")

patients = df["Patient ID"].unique()
np.random.shuffle(patients)

train_split = int(0.7 * len(patients))
val_split = int(0.85 * len(patients))

train_ids = patients[:train_split]
val_ids = patients[train_split:val_split]
test_ids = patients[val_split:]

os.makedirs("data/splits", exist_ok=True)

with open("data/splits/train_patients.txt", "w") as f:
    for pid in train_ids:
        f.write(str(pid) + "\n")

with open("data/splits/val_patients.txt", "w") as f:
    for pid in val_ids:
        f.write(str(pid) + "\n")

with open("data/splits/test_patients.txt", "w") as f:
    for pid in test_ids:
        f.write(str(pid) + "\n")

print("Splits created successfully!")