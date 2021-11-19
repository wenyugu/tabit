import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

DATA_PATH = '../spec_repr/'

class GuitarSetDataset(Dataset):
    def __init__(self, partition_ids, data_dir=DATA_PATH, context_win_size=9, spec_mode='c'):
        self.data_dir = data_dir
        self.context_win_size = context_win_size
        self.partitiion_ids = partition_ids
        self.spec_mode = spec_mode
        self.halfwin = context_win_size // 2


    def __len__(self):
        return len(self.partitiion_ids)

    def __getitem__(self, idx):
        
        path = self.data_dir + self.spec_mode + "/"
        frame = self.partitiion_ids[idx]
        filename = "_".join(frame.split("_")[:-1]) + ".npz"
        frame_idx = int(frame.split("_")[-1])
        
        # load a context window centered around the frame index
        loaded = np.load(path + filename)
        full_x = np.pad(loaded["repr"], [(self.halfwin, self.halfwin), (0,0)], mode='constant')
        sample_x = full_x[frame_idx:frame_idx + self.context_win_size]
        return np.swapaxes(sample_x, 0, 1), loaded["labels"][frame_idx]

if __name__ == '__main__':
    partition_csv = './id.csv'
    train_partition = []
    test_partition = []
    k = 0 # fold index
    for item in list(pd.read_csv(partition_csv, header=None)[0]):
        fold = int(item.split("_")[0])
        if fold == k:
            test_partition.append(item)
        else:
            train_partition.append(item)

    train_dataset = GuitarSetDataset(train_partition)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True)
    # TODO: set pin_memeory to True for GPU

    print(len(train_dataset), len(train_loader))
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        break

    test_dataset = GuitarSetDataset(test_partition)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False)
    print(len(test_dataset), len(test_loader))
    for batch_idx, data in enumerate(test_loader):
        inputs, labels = data
        print(inputs.shape, labels.shape)
        break
