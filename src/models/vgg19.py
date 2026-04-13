# models/vgg19.py

import torch.nn as nn
from torchvision.models import vgg19

def get_vgg19():
    model = vgg19(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 14)
    return model
