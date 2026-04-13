# models/resnet18.py

import torch.nn as nn
from torchvision.models import models

def get_resnet18(num_classes=14):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
