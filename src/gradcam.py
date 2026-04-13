# src/gradcam.py

import torch
import cv2
import numpy as np

def generate_gradcam(model, image, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image)
    output.max().backward()

    grad = gradients[0].mean(dim=(2,3), keepdim=True)
    cam = (grad * activations[0]).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    return cam
