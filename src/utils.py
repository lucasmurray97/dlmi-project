from torch.autograd import Function
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
    
class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        with h5py.File(self.dataset_path, 'r') as hdf:        
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img_np = np.array(hdf.get(img_id).get('img'))
            img = torch.tensor(img_np)
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
            center = np.array(hdf.get(img_id).get('metadata'))[0] if self.mode == 'train' else None
        return self.preprocessing(img).float(), label, center
    
class ValBaselineDataset(BaselineDataset):
    def __init__(self, dataset_path, preprocessing):
        super(ValBaselineDataset, self).__init__(dataset_path, preprocessing, mode='test')
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img_np = np.array(hdf.get(img_id).get('img'))
            img = torch.tensor(img_np)
            label = np.array(hdf.get(img_id).get('label'))
        return self.preprocessing(img).float(), label, torch.tensor(3)

class TestBaselineDataset(BaselineDataset):
    def __init__(self, dataset_path, preprocessing):
        super(TestBaselineDataset, self).__init__(dataset_path, preprocessing, mode='test')
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img_np = np.array(hdf.get(img_id).get('img'))
            img = torch.tensor(img_np)
        return self.preprocessing(img).float(), torch.tensor(4)
    
def init_weights_xavier(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class AlphaScheduler:
    def __init__(self, total_steps, max_alpha=1.0, min_alpha=0.0, gamma=10.0):
        self.total_steps = total_steps
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.current_step = 0

    def step(self):
        """Advance one step and return updated alpha."""
        p = self.current_step / self.total_steps
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)
        self.current_step += 1
        return alpha
