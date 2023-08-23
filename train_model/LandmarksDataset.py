import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HandLandmarksDataset(Dataset):
    """Hand Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.landmarks_frame.iloc[idx, 60]
        landmarks = self.landmarks_frame.iloc[idx, :-1]
        landmarks = np.array([landmarks], dtype=float).reshape(20, 3)
        item = {'labels': label, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return item
