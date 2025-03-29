import h5py
import torch
import random
import numpy as np
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import json

# Get current path
import os
import sys
from pathlib import Path
import os

current_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'train.h5'))
VAL_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'val.h5'))
TEST_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'test.h5'))
SEED = 0
from utils import BaselineDataset, TestBaselineDataset, ValBaselineDataset
from models import DANN


# Load data
train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]), mode='train')
val_dataset = ValBaselineDataset(VAL_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataset = TestBaselineDataset(TEST_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)), transforms.ToTensor()]))

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# Define model
model = DANN(input_dim=512, hidden_dim=256, num_classes=2, lambda_grl=1.0, num_dom=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_class_accuracy = torchmetrics.Accuracy("binary").to(device)
val_class_accuracy = torchmetrics.Accuracy("binary").to(device)
train_domain_accuracy = torchmetrics.Accuracy("multiclass", num_classes=5).to(device)
val_domain_accuracy = torchmetrics.Accuracy("multiclass", num_classes=5).to(device)
test_domain_accuracy = torchmetrics.Accuracy("multiclass", num_classes=5).to(device)
train_loss = []
val_loss = []
test_loss = []
train_class_acc_h = []
train_dom_acc_h = []
val_class_acc_h = []
val_dom_acc_h = []
test_domain_acc_h = []

# Define training parameters
n_epoch = 5
alpha = 1.
len_dataloader = len(train_dataloader)
model.to(device)

# Training loop
best_val_acc = 0
for epoch in tqdm(range(n_epoch)):
    model.train()
    train_ep_loss = 0
    val_ep_loss = 0
    test_ep_loss = 0
    train_class_accuracy.reset()
    train_domain_accuracy.reset()
    for i, (x, y, c) in enumerate(tqdm(train_dataloader)):
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        label_output = model.label_classifier(features)
        domain_output = model.domain_discriminator(model.grl(features))
        label_loss = criterion(label_output, y)
        domain_loss = criterion(domain_output, c)
        loss = label_loss + domain_loss
        loss.backward()
        optimizer.step()
        probs_class = F.softmax(label_output, dim=1)
        probs_domain = F.softmax(domain_output, dim=1)
        train_class_accuracy(torch.argmax(probs_class, dim=1), y)
        train_domain_accuracy(torch.argmax(probs_domain, dim=1), c)
        train_ep_loss += loss.item()
    train_loss.append(train_ep_loss / len(train_dataloader))
    train_class_acc_h.append(train_class_accuracy.compute().item())
    train_dom_acc_h.append(train_domain_accuracy.compute().item())
    print(f'Epoch {epoch+1}/{n_epoch} - Train Accuracy: {train_class_accuracy.compute()} - Train Domain Accuracy: {train_domain_accuracy.compute()}')
    val_class_accuracy.reset()
    val_domain_accuracy.reset()
    for i, (x, y, c) in enumerate(tqdm(val_dataloader)):
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        label_output = model.label_classifier(features)
        domain_output = model.domain_discriminator(model.grl(features))
        label_loss = criterion(label_output, y)
        domain_loss = criterion(domain_output, c)
        loss = domain_loss
        loss.backward()
        optimizer.step()
        probs_class = F.softmax(label_output, dim=1)
        probs_domain = F.softmax(domain_output, dim=1)
        val_class_accuracy(torch.argmax(probs_class, dim=1), y)
        val_domain_accuracy(torch.argmax(probs_domain, dim=1), c)
        val_ep_loss += loss.item()
    val_loss.append(val_ep_loss / len(val_dataloader))
    val_class_acc_h.append(val_class_accuracy.compute().item())
    val_dom_acc_h.append(val_domain_accuracy.compute().item())
    print(f'Epoch {epoch+1}/{n_epoch} - Val Accuracy: {val_class_accuracy.compute()} - Val Domain Accuracy: {val_domain_accuracy.compute()}')
    test_domain_accuracy.reset()
    for i, (x, c) in enumerate(tqdm(test_dataloader)):
        x, c = x.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        domain_output = model.domain_discriminator(model.grl(features))
        domain_loss = criterion(domain_output, c)
        loss = domain_loss
        loss.backward()
        optimizer.step()
        probs_domain = F.softmax(domain_output, dim=1)
        test_domain_accuracy(torch.argmax(probs_domain, dim=1), c)
        test_ep_loss += loss.item()
    test_loss.append(test_ep_loss / len(test_dataloader))
    test_domain_acc_h.append(test_domain_accuracy.compute().item())
    print(f'Epoch {epoch+1}/{n_epoch} - Test Domain Accuracy: {test_domain_accuracy.compute()}')

    # Save model if val_class_accuracy is better
    if val_class_accuracy.compute() > best_val_acc:
        best_val_acc = val_class_accuracy.compute()
        torch.save(model.state_dict(), 'model.pth')
        print('Model saved')

# Plot results
plt.title('Losses')
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.savefig('loss.png')
# clear
plt.clf()

# Plot class accuracies
plt.title('Class Accuracies')
plt.plot(train_class_acc_h, label='Train Class Accuracy')
plt.plot(val_class_acc_h, label='Val Class Accuracy')
plt.legend()
plt.savefig('class_acc.png')
# clear
plt.clf()

# Plot domain accs
plt.title('Domain Accuracy')
plt.plot(train_dom_acc_h, label='Train Domain Accuracy')
plt.plot(val_dom_acc_h, label='Val Domain Accuracy')
plt.plot(test_domain_acc_h, label='Test Domain Accuracy')
plt.legend()
plt.savefig('dom_acc.png')
# clear
plt.clf()


# Save results in json
results = {
    'train_class_accuracy': train_class_accuracy.compute().item(),
    'val_class_accuracy': val_class_accuracy.compute().item(),
    'train_domain_accuracy': train_domain_accuracy.compute().item(),
    'val_domain_accuracy': val_domain_accuracy.compute().item(),
    'test_domain_accuracy': test_domain_accuracy.compute().item()
}

# save results to json
with open('results.json', 'w') as f:
    json.dump(results, f)

