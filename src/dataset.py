import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

class WeightHeightDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X = self.data.iloc[:, 1].values.reshape(-1, 1)  # Height (input)
        self.Y = self.data.iloc[:, 2].values.reshape(-1, 1)  # Weight (output)

        # Normalize Height (input) between 0-1
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

def load_data(file_path, batch_size=32, train_split=0.7, val_split=0.15):
    logging.debug(f"📂 Loading dataset from: {file_path}")

    dataset = WeightHeightDataset(file_path)
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    logging.debug(f"✅ Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
