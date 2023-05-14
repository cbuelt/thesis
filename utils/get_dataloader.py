import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader



def get_data_loader(batch_size):
    dataset = SolarImage(
        data_path="../data/",
        ts_path="../data/ts/full_data.pkl",
        index_file="../data/index.csv",
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader, dataset


class SpatialField(Dataset):
    def __init__(
        self,
        data_path,
        train,
    ):
        self.data_path = data_path
        self.train = train
        self.img_data = np.load(data_path)
        self.param_data = np.load(data_path)
        self.sample_size = len(param_data)


    def __len__(self):
        return len(self.sample_size)

    def __getitem__(self, idx):
        img = self.img_data[idx]
        param = self.param_data[idx]

        return img, param


#dataloader, dataset = get_data_loader(batch_size=4)
#sample = dataset.__getitem__(10)
#ghi = sample["ghi"]
#print(ghi)
