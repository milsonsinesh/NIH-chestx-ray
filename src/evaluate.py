# src/evaluate.py

import torch
import numpy as np

def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = torch.sigmoid(model(images))
            preds.append(outputs.cpu().numpy())
            targets.append(labels.numpy())

    return np.vstack(preds), np.vstack(targets)
