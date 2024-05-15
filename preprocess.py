# torch preprocessing.
# torch preprocessing might look simple because the data is already standardized and spit,
# so we only need to load the right file.
# Full TensorFlow preprocessing with split batches is in the tf_preprocess directory.
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class AnomalyDetectionDataset(Dataset):
    # Read data as a pandas DataFrame and return as a torch tensor.
    def __init__(self, file_path, file_csv=True):
        self.data = pd.read_csv(file_path) if file_csv else pd.read_hdf(file_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        item = self.data.iloc[ind].to_numpy(dtype='float32')
        return torch.from_numpy(item)

def load_anomaly_dataset_csv(file_name, file_directory, batch_size):
    # Load data as a pandas DataFrame and return as torch.DataLoader.
    datasets = {}
    for ind, (key, value) in enumerate(file_name.items()):
        dataset = AnomalyDetectionDataset(f'{file_directory}/{value}.csv')
        torch.manual_seed(42)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datasets.update({key: dataloader})
    return datasets

def device_change():
    # Switch to GPU if it is available.
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    return device
