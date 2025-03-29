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
        # return 200

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
            center = np.array(hdf.get(img_id).get('metadata'))[0] if self.mode == 'train' else None
        return self.preprocessing(img).float(), label, center
    
class ValBaselineDataset(BaselineDataset):
    def __init__(self, dataset_path, preprocessing):
        super(ValBaselineDataset, self).__init__(dataset_path, preprocessing, mode='test')
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
            label = np.array(hdf.get(img_id).get('label'))
        return self.preprocessing(img).float(), label, torch.tensor(3)

class TestBaselineDataset(BaselineDataset):
    def __init__(self, dataset_path, preprocessing):
        super(TestBaselineDataset, self).__init__(dataset_path, preprocessing, mode='test')
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
        return self.preprocessing(img).float(), torch.tensor(4)