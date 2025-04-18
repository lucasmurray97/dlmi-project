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
import torch.optim as optim
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import json
from utils_training import AlphaScheduler

# Get current path
import os
import sys
from pathlib import Path
from utils_training import BaselineDataset, TestBaselineDataset, ValBaselineDataset
from models import DANN
import dino_mod.vision_transformer as vits
from dino_mod.utils import restart_from_checkpoint
import argparse

current_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'train.h5'))
VAL_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'val.h5'))
TEST_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'test.h5'))
SEED = 0


# Add argument parser
parser = argparse.ArgumentParser(description='Train DINO model')
parser.add_argument('--weights', type=str, help='Train the model')
# batch size
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
# num workers
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers')
# num epochs
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
# experiment name
parser.add_argument('--experiment_name', type=str, default='dino', help='Experiment name')
# patch size
parser.add_argument('--patch_size', type=int, default=8, help='Patch size')
# Number of layers
parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
# Number of hidden dimensions
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimensions')
# List of layers to be unfrozen
parser.add_argument('--unfreeze_layers', type=str, default='', help='Layers to be unfrozen')
args = parser.parse_args()

# Load data
train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, mode='train')
val_dataset = ValBaselineDataset(VAL_IMAGES_PATH)
test_dataset = TestBaselineDataset(TEST_IMAGES_PATH)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Define model
backbone = vits.__dict__['vit_small'](
            patch_size=8,
            drop_path_rate=0.1,  # stochastic depth
        )
# current path
current_path = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_path, f'./weights/pre-training/{args.weights}')

restart_from_checkpoint(
        weights_path,
        student=backbone,
    )

model = DANN(input_dim=512, hidden_dim=args.hidden_dim, num_classes=2, lambda_grl=1.0, num_dom=5, backbone=backbone, num_hidden_layers=args.num_layers)

# Unfreeze layers
if args.unfreeze_layers != '':
    unfreeze_layers = [int(x) for x in args.unfreeze_layers.split(',')]
    for name, param in model.feature_extractor.named_parameters():
        if any(f'blocks.{layer}' in name for layer in unfreeze_layers):
            param.requires_grad = True
        elif 'norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# print number of trainable parameters 
print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print(len(train_dataset))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

# Define metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_class_accuracy = torchmetrics.Accuracy("binary").to(device)
val_class_accuracy = torchmetrics.Accuracy("binary").to(device)
train_loss = []
val_loss = []
test_loss = []
train_class_acc_h = []
val_class_acc_h = []

# Define training parameters
n_epoch = args.num_epochs
model.to(device)

# Training loop
best_val_acc = 0
steps_per_epoch = len(train_dataloader) + len(val_dataloader) + len(test_dataloader)
total_steps = n_epoch * steps_per_epoch

for epoch in tqdm(range(n_epoch)):
    model.train()
    train_ep_loss = 0
    val_ep_loss = 0
    test_ep_loss = 0
    train_class_accuracy.reset()
    ## Train ###
    for i, (x, y, c) in enumerate(train_dataloader):
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        label_output = model.label_classifier(features)
        label_loss = criterion(label_output, y)
        loss = label_loss
        loss.backward()
        optimizer.step()
        probs_class = F.softmax(label_output, dim=1)
        train_class_accuracy(torch.argmax(probs_class, dim=1), y)
        train_ep_loss += loss.item()
    train_loss.append(train_ep_loss / len(train_dataloader))
    train_class_acc_h.append(train_class_accuracy.compute().item())
    print(f'Epoch {epoch+1}/{n_epoch} - Train Accuracy: {train_class_accuracy.compute()}')
    
    ### Val ###
    val_class_accuracy.reset()
    for i, (x, y, c) in enumerate(val_dataloader):
        with torch.no_grad():
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad()
            features = model.feature_extractor(x)
            label_output = model.label_classifier(features)
            label_loss = criterion(label_output, y)
            loss = label_loss
            probs_class = F.softmax(label_output, dim=1)
            val_class_accuracy(torch.argmax(probs_class, dim=1), y)
            val_ep_loss += loss.item()
    val_loss.append(val_ep_loss / len(val_dataloader))
    val_class_acc_h.append(val_class_accuracy.compute().item())
    print(f'Epoch {epoch+1}/{n_epoch} - Val Accuracy: {val_class_accuracy.compute()}')
    
    scheduler.step()
    # Save model if val_class_accuracy is better
    if val_class_accuracy.compute() > best_val_acc:
        best_val_acc = val_class_accuracy.compute()
        torch.save(model.state_dict(), f'model_{args.experiment_name}.pth')
        print('Model saved')

# Plot results
plt.title('Losses')
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.legend()
plt.savefig(f'figures/loss_dino_{args.experiment_name}.png')
# clear
plt.clf()

# Plot class accuracies
plt.title('Class Accuracies')
plt.plot(train_class_acc_h, label='Train Class Accuracy')
plt.plot(val_class_acc_h, label='Val Class Accuracy')
plt.legend()
plt.savefig(f'figures/class_acc_dino_{args.experiment_name}.png')
# clear
plt.clf()



# Save results in json
results = {
    'train_class_accuracy': train_class_accuracy.compute().item(),
    'val_class_accuracy': best_val_acc.item(),
}

# save results to json
with open(f'results_{args.experiment_name}.json', 'w') as f:
    json.dump(results, f)

