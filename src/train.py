# src/train.py

import torch
from torch.optim import Adam
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
