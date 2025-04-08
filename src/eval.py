
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
import os
from utils_training import BaselineDataset, TestBaselineDataset, ValBaselineDataset
from models import DANN
import dino_mod.vision_transformer as vits
from dino_mod.utils import restart_from_checkpoint

current_path = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'test.h5'))
TRAIN_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'train.h5'))
VAL_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'val.h5'))
SEED = 0

# Define model
backbone = vits.__dict__['vit_small'](
            patch_size=14,
            drop_path_rate=0.1,  # stochastic depth
        )
        

restart_from_checkpoint(
        os.path.join('./weights/pre-training/', "dino_14.pth"),
        student=backbone,
    )

model = DANN(input_dim=512, hidden_dim=256, num_classes=2, lambda_grl=1.0, num_dom=5, backbone=backbone)

state_dict = torch.load('./src/weights/training/model_dino_14.pth', map_location='cpu')
model.load_state_dict(state_dict)

# Carga solo en el backbone DINO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

with h5py.File(VAL_IMAGES_PATH, 'r') as hdf:
    val_ids = list(hdf.keys())

tf = transforms.Compose([transforms.ToPILImage(),transforms.Resize((98, 98)), transforms.ToTensor()])
acc = 0
with h5py.File(VAL_IMAGES_PATH, 'r') as hdf:
    for val_id in tqdm(val_ids[:1000]):
        img = tf(torch.tensor(np.array(hdf.get(val_id).get('img')))).unsqueeze(0).float().to(device)
        class_ = np.array(hdf.get(val_id).get('label'))
        features = model.feature_extractor(img)
        label_output = model.label_classifier(features)
        class_pred = torch.argmax(label_output).detach().cpu()
        acc += class_pred.item() == class_
print("Accuracy: ", acc / len(val_ids[:1000]))

with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    test_ids = list(hdf.keys())

solutions_data = {'ID': [], 'Pred': []}
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    for test_id in tqdm(test_ids):
        img = tf(torch.tensor(np.array(hdf.get(test_id).get('img')))).unsqueeze(0).float().to(device)
        features = model.feature_extractor(img)
        label_output = model.label_classifier(features)
        class_pred = torch.argmax(label_output).detach().cpu()
        solutions_data['ID'].append(int(test_id))
        solutions_data['Pred'].append(class_pred.item())

solutions_data = pd.DataFrame(solutions_data).set_index('ID')
solutions_data.to_csv('submission.csv')