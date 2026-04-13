# src/train_with_tracking.py

import os
import torch
import mlflow
import mlflow.pytorch
import wandb
import numpy as np

from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import *
from src.losses import get_loss
from src.dataset import ChestXrayDataset
from src.transforms import train_transforms, val_transforms

from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------
# CONFIG
# ---------------------------
USE_WANDB = True
EXPERIMENT_NAME = "NIH-ChestXray-ResNet18"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# UTILS
# ---------------------------
def compute_mean_auroc(y_true, y_pred):

    aurocs = []

    for i in range(y_true.shape[1]):

        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aurocs.append(score)

        except ValueError:
            continue

    if len(aurocs) == 0:
        return 0.0

    return np.mean(aurocs)


# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer):

    model.train()
    running_loss = 0

    for images, labels in tqdm(loader, desc="Training"):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ---------------------------
# VALIDATION FUNCTION
# ---------------------------
def validate(model, loader):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Validation"):

            images = images.to(DEVICE)

            outputs = torch.sigmoid(model(images))

            preds.append(outputs.cpu().numpy())
            targets.append(labels.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    mean_auroc = compute_mean_auroc(targets, preds)

    return mean_auroc


# ---------------------------
# MAIN TRAINING PIPELINE
# ---------------------------
def main():

    # -------- MLflow setup --------
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        mlflow.log_params({
            "model": "ResNet18",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "image_size": IMAGE_SIZE,
            "loss": "BCEWithLogitsLoss"
        })


        # -------- W&B setup --------
        if USE_WANDB:

            wandb.init(
                project="NIH-ChestXray14",
                name=EXPERIMENT_NAME,
                config={
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR
                }
            )


        # -------- Dataset --------
        train_ids = open("data/splits/train_patients.txt").read().splitlines()
        val_ids = open("data/splits/val_patients.txt").read().splitlines()

        train_ds = ChestXrayDataset(
            csv_path="data/raw/Data_Entry_2017.csv",
            image_dir="data/raw/images",
            patient_ids=train_ids,
            transform=train_transforms()
        )

        val_ds = ChestXrayDataset(
            csv_path="data/raw/Data_Entry_2017.csv",
            image_dir="data/raw/images",
            patient_ids=val_ids,
            transform=val_transforms()
        )

        print("Train dataset size:", len(train_ds))
        print("Val dataset size:", len(val_ds))


        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )


        # -------- Model --------
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace final layer (1000 -> 14)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

        model = model.to(DEVICE)


        # -------- Loss & Optimizer --------
        class_weights = torch.ones(NUM_CLASSES).to(DEVICE)

        criterion = get_loss(class_weights)

        optimizer = Adam(model.parameters(), lr=LR)


        best_auroc = 0


        # -------- Training Loop --------
        for epoch in range(1, EPOCHS + 1):

            print(f"\nEpoch {epoch}/{EPOCHS}")

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_auroc = validate(model, val_loader)


            # -------- Logging --------
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_mean_AUROC", val_auroc, step=epoch)

            if USE_WANDB:

                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_mean_AUROC": val_auroc
                })


            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Mean AUROC: {val_auroc:.4f}")


            # -------- Save Best Model --------
            if np.isnan(val_auroc) or val_auroc > best_auroc:
                
             best_auroc = val_auroc
            torch.save(model.state_dict(), "best_model.pt")
            print("✅ Best model saved")
            
            torch.save(model.state_dict(), "last_model.pt")
            print("✅ Last model saved")


        # -------- Log Final Model --------
        mlflow.pytorch.log_model(model, artifact_path="model")


        if USE_WANDB:
            wandb.finish()


        print("\n🎉 Training completed")
        print(f"Best Validation AUROC: {best_auroc:.4f}")


# ---------------------------
if __name__ == "__main__":
    main()