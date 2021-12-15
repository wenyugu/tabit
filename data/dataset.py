import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

DATA_PATH = os.path.abspath(os.path.dirname(__file__)) + '/spec_repr/'

class GuitarSetDataset(Dataset):
    def __init__(self, partition_ids, data_dir=DATA_PATH, context_win_size=9, spec_mode='c', seq2seq=False):
        self.data_dir = data_dir
        self.context_win_size = context_win_size
        self.partition_ids = partition_ids
        self.spec_mode = spec_mode
        self.halfwin = context_win_size // 2
        self.seq2seq = seq2seq
        if self.seq2seq:
            self.seqs = []
            # list of (filename, start_frame_idx)
            for partition_id in sorted(partition_ids):
                frame_idx = int(partition_id.split("_")[-1])
                filename = "_".join(partition_id.split("_")[:-1]) + ".npz"
                if frame_idx % self.halfwin == 0:
                    self.seqs.append((filename, frame_idx))


    def __len__(self):
        if self.seq2seq:
            return len(self.seqs)
        return len(self.partition_ids)

    def __getitem__(self, idx):
        path = self.data_dir + self.spec_mode + "/"
        if self.seq2seq:
            seq = self.seqs[idx]
            filename, start_idx = seq
            loaded = np.load(path + filename)
            features = loaded['repr'][start_idx:start_idx + self.context_win_size]
            labels = loaded['labels'][start_idx:start_idx + self.context_win_size]
            if len(features) < self.context_win_size:
                X = torch.from_numpy(np.pad(features, [(0, self.context_win_size - len(features)), (0, 0)])).float()
                closed = np.zeros((self.context_win_size - len(labels), 6, 21))
                closed[:, :, 0] = 1
                y = torch.from_numpy(np.vstack((labels, closed))).float()
                assert X.size() == torch.Size([self.context_win_size, features.shape[1]])
                assert y.size() == torch.Size([self.context_win_size, 6, 21])
                return X, y
            
            X = torch.from_numpy(features).float()
            y = torch.from_numpy(labels).float()
            assert X.size() == torch.Size([self.context_win_size, features.shape[1]])
            assert y.size() == torch.Size([self.context_win_size, 6, 21])
            return X, y
        
        
        frame = self.partition_ids[idx]
        filename = "_".join(frame.split("_")[:-1]) + ".npz"
        frame_idx = int(frame.split("_")[-1])
        
        # load a context window centered around the frame index
        loaded = np.load(path + filename)
        full_x = np.pad(loaded["repr"], [(self.halfwin, self.halfwin), (0,0)], mode='constant')
        sample_x = full_x[frame_idx:frame_idx + self.context_win_size]
        X = torch.from_numpy(np.swapaxes(sample_x, 0, 1)[None, :]).float()
        y = torch.from_numpy(loaded["labels"][frame_idx]).float()
        return X, y

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
