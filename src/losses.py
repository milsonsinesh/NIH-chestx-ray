# src/losses.py

import torch.nn as nn

def get_loss(class_weights):
    return nn.BCEWithLogitsLoss(pos_weight=class_weights)
