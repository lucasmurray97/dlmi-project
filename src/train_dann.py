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
import os
import sys
from pathlib import Path
import os
from utils_training import BaselineDataset, TestBaselineDataset, ValBaselineDataset
from models import DANN

current_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'train.h5'))
VAL_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'val.h5'))
TEST_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'test.h5'))
SEED = 0



# Load data
train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(), transforms.Resize((98, 98)), transforms.ToTensor()]), mode='train')
val_dataset = ValBaselineDataset(VAL_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(),transforms.Resize((98, 98)), transforms.ToTensor()]))
test_dataset = TestBaselineDataset(TEST_IMAGES_PATH, transforms.Compose([transforms.ToPILImage(),transforms.Resize((98, 98)), transforms.ToTensor()]))

train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=16)

# Define model
model = DANN(input_dim=512, hidden_dim=256, num_classes=2, lambda_grl=1.0, num_dom=5)
# print number of trainable parameters 
print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

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
n_epoch = 50
alpha = 1.
len_dataloader = len(train_dataloader)
model.to(device)

# Training loop
best_val_acc = 0
steps_per_epoch = len(train_dataloader) + len(val_dataloader) + len(test_dataloader)
total_steps = n_epoch * steps_per_epoch
alpha_scheduler = AlphaScheduler(total_steps)

for epoch in tqdm(range(n_epoch)):
    model.train()
    train_ep_loss = 0
    val_ep_loss = 0
    test_ep_loss = 0
    train_class_accuracy.reset()
    train_domain_accuracy.reset()
    ## Train ###
    for i, (x, y, c) in enumerate(train_dataloader):
        alpha = alpha_scheduler.step()
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        label_output = model.label_classifier(features)
        domain_output = model.domain_discriminator(model.grl(features, alpha))
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
    print("Alpha: ", alpha) 
    
    ### Val ###
    val_class_accuracy.reset()
    val_domain_accuracy.reset()
    for i, (x, y, c) in enumerate(val_dataloader):
        alpha = alpha_scheduler.step()
        x, y, c = x.to(device), y.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        label_output = model.label_classifier(features)
        domain_output = model.domain_discriminator(model.grl(features, alpha))
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
    print("Alpha: ", alpha) 
    ### Test ###
    test_domain_accuracy.reset()
    for i, (x, c) in enumerate(test_dataloader):
        alpha = alpha_scheduler.step()
        x, c = x.to(device), c.to(device)
        optimizer.zero_grad()
        features = model.feature_extractor(x)
        domain_output = model.domain_discriminator(model.grl(features, alpha))
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
    print("Alpha: ", alpha) 
    scheduler.step()
    # Save model if val_class_accuracy is better
    if val_class_accuracy.compute() > best_val_acc:
        best_val_acc = val_class_accuracy.compute()
        torch.save(model.state_dict(), 'model_dann.pth')
        print('Model saved')

# Plot results
plt.title('Losses')
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.savefig('loss_dann.png')
# clear
plt.clf()

# Plot class accuracies
plt.title('Class Accuracies')
plt.plot(train_class_acc_h, label='Train Class Accuracy')
plt.plot(val_class_acc_h, label='Val Class Accuracy')
plt.legend()
plt.savefig('class_acc_dann.png')
# clear
plt.clf()

# Plot domain accs
plt.title('Domain Accuracy')
plt.plot(train_dom_acc_h, label='Train Domain Accuracy')
plt.plot(val_dom_acc_h, label='Val Domain Accuracy')
plt.plot(test_domain_acc_h, label='Test Domain Accuracy')
plt.legend()
plt.savefig('dom_acc_dann.png')
# clear
plt.clf()


# Save results in json
results = {
    'train_class_accuracy': train_class_accuracy.compute().item(),
    'val_class_accuracy': best_val_acc.item(),
    'train_domain_accuracy': train_domain_accuracy.compute().item(),
    'val_domain_accuracy': val_domain_accuracy.compute().item(),
    'test_domain_accuracy': test_domain_accuracy.compute().item()
}

# save results to json
with open('results_dann.json', 'w') as f:
    json.dump(results, f)

