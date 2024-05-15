import torch
from torch.utils.data import Dataset, DataLoader
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
        # item = np.array([(x, x) for x in item])
        if self.three_channels:
            item = np.stack([item[:700], item[700:1400], item[-700:]], axis=0)
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


def load_anomaly_dataset_csv(file_name, file_directory, batch_size, three_channels=True):
    datasets = {}
    for ind, (key, value) in enumerate(file_name.items()):
        dataset = AnomalyDetectionDataset(f'{file_directory}/{value}.csv', three_channels=three_channels)
        torch.manual_seed(42)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datasets.update({key: dataloader})
    return datasets