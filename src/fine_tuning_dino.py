import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import math
import os
from utils import CombinedH5UnlabeledDataset
from tqdm import tqdm

from dino import DinoLoss, DINOHead, MultiCropWrapper  # Deben estar implementados

# ============================
# Configuración
# ============================
class Config:
    batch_size = 1024
    num_workers = 16
    epochs = 100
    base_lr = 1e-4
    out_dim = 65536
    student_temp = 0.1
    teacher_temp = 0.04
    warmup_teacher_temp = 0.04
    warmup_teacher_temp_epochs = 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()


# ============================
# Transformaciones
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ============================
# Momentum update del teacher
# ============================

# ============================
# Cargar datos y modelo
# ============================
current_path = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'train.h5'))
VAL_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'val.h5'))
TEST_IMAGES_PATH = os.path.abspath(os.path.join(current_path, '..', 'data', 'test.h5'))
h5_paths = [TRAIN_IMAGES_PATH, VAL_IMAGES_PATH, TEST_IMAGES_PATH]
train_dataset = CombinedH5UnlabeledDataset(h5_paths, transform=transform)

dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
backbone.eval()
embed_dim = backbone.embed_dim
backbone.to(cfg.device)

student_head = DINOHead(embed_dim, cfg.out_dim)
teacher_head = DINOHead(embed_dim, cfg.out_dim)

student = MultiCropWrapper(backbone, student_head).to(cfg.device)
teacher = MultiCropWrapper(backbone, teacher_head).to(cfg.device)

#print number of parameters
print(f"Number of trainable parameters: {sum(p.numel() for p in student.parameters() if p.requires_grad)}")

teacher.load_state_dict(student.state_dict())
for p in teacher.parameters():
    p.requires_grad = False

# ============================
# Pérdida y optimizador
# ============================
dino_loss = DinoLoss(
    out_dim=cfg.out_dim,
    student_temp=cfg.student_temp,
    teacher_temp=cfg.teacher_temp,
    warmup_teacher_temp=cfg.warmup_teacher_temp,
    warmup_teacher_temp_epochs=cfg.warmup_teacher_temp_epochs,
    nepochs=cfg.epochs,
    ncrops=2  # ← this is what was missing
).to(cfg.device)


optimizer = optim.AdamW(student.parameters(), lr=cfg.base_lr)

# ============================
# Entrenamiento
# ============================
for epoch in tqdm(range(cfg.epochs)):
    student.train()
    total_loss = 0


    for images, _ in dataloader:
        images = images.to(cfg.device)

        teacher_output = teacher(images)
        student_output = student(images)

        loss = dino_loss(student_output, teacher_output, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss / len(dataloader):.4f}")

# ============================
# Guardar modelo
# ============================
torch.save(student.state_dict(), "dino_student.pth")