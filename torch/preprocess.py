import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class AnomalyDetectionDataset(Dataset):
    def __init__(self, file_path, three_channels=True, file_csv=True):
        self.data = pd.read_csv(file_path) if file_csv else pd.read_hdf(file_path)
        self.three_channels = three_channels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        item = self.data.iloc[ind].to_numpy(dtype='float32')
        tuple_item = lambda x: (x, x)
        item = np.array([tuple_item(i) for i in item])
        if self.three_channels:
            item = np.stack([item[:700], item[700:1400], item[-700:]], axis=0)
        print(f'Generated tensor shape: {item.shape}')
        return torch.from_numpy(item)

def device_change():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    return device