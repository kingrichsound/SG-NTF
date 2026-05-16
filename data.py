"""
Dataloading module
"""

import random

import torch
from torch.utils.data import Dataset

random.seed(0)


class TensorDataset(Dataset):
    """Wrapper for a 3D HDI tensor dataset with A, B, C indices and values"""

    def __init__(self, aIDs, bIDs, cIDs, values) -> None:
        self.A_Indices = aIDs
        self.B_Indices = bIDs
        self.C_Indices = cIDs
        self.values = values

    def __getitem__(self, index):
        return (
            self.A_Indices[index],
            self.B_Indices[index],
            self.C_Indices[index],
            self.values[index],
        )

    def __len__(self):
        return self.A_Indices.size(0)

    def __repr__(self) -> str:
        return (
            f"QoSDataset(users={self.A_Indices}, services={self.B_Indices},"
            f" timeslices={self.C_Indices}, values={self.values})"
        )

    def log_transform(self):
        """Log transformation"""
        self.values = torch.log(self.values)

    def inv_log_transform(self):
        """Inverse log transformation"""
        self.values = torch.exp(self.values)

    def minmax_norm(self, min, max):
        """Min-max normalization"""
        self.values = (self.values - min) / (max - min)

    def zscore_norm(self, mean, std):
        """Z-score normalization"""
        self.values = (self.values - mean) / 2 * std + 0.5

    def minmax_denorm(self, min, max):
        """Min-max denormalization"""
        self.values = self.values * (max - min) + min

    def zscore_denorm(self, mean, std):
        """Z-score denormalization"""
        self.values = (self.values - 0.5) * 2 * std + mean


def load(filepath: str):
    """Load data from a file"""
    A_Indices = []
    B_Indices = []
    C_Indices = []
    values = []

    # process data line by line
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            user = data[0]
            service = data[1]
            time = data[2]
            value = data[3]
            A_Indices.append(int(user))
            B_Indices.append(int(service))
            C_Indices.append(int(time))
            values.append(float(value))

    A_Indices = torch.tensor(A_Indices, dtype=torch.int)
    B_Indices = torch.tensor(B_Indices, dtype=torch.int)
    C_Indices = torch.tensor(C_Indices, dtype=torch.int)
    values = torch.tensor(values, dtype=torch.float)
    return TensorDataset(A_Indices, B_Indices, C_Indices, values)
