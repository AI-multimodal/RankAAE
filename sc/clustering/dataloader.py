from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import pandas as pd
import math
import numpy as np
from torchvision import transforms


class CoordNumSpectraDataset(Dataset):
    def __init__(self, csv_fn, split_portion, train_val_test_ratios=(0.7, 0.15, 0.15), sampling_exponent=0.6,
                 n_coord_num=3, transform=None):
        full_df = pd.read_csv(csv_fn, index_col=[0, 1])
        n_train_val_test = [int(len(full_df) * ratio) for ratio in train_val_test_ratios]
        n_train_val_test[-1] = int(len(full_df)) - sum(n_train_val_test[:-1])
        portion_options = ['train', 'val', 'test']
        assert split_portion in portion_options
        i_prev = portion_options.index(split_portion)
        df = full_df[sum(n_train_val_test[:i_prev]):sum(n_train_val_test[:i_prev+1])]
        assert "ENE_" in df.columns.to_list()[n_coord_num]
        assert "ENE_" not in df.columns.to_list()[n_coord_num-1]
        data = df.to_numpy()
        self.cn = data[:, :n_coord_num]
        self.spec = data[:, n_coord_num:]
        self.transform = transform
        self.atom_index = df.index.to_list()

        train_df = full_df[:n_train_val_test[0]]
        sampling_weights_per_cn = 1.0 / np.fabs(train_df.to_numpy()[:, :n_coord_num]).mean(axis=0)
        sampling_weights_per_cn **= sampling_exponent
        sampling_weights_per_cn /= sampling_weights_per_cn.sum()
        self.sampling_weights_per_cn = sampling_weights_per_cn

        self.sampling_weights_per_sample = self.sampling_weights_per_cn[np.argmax(np.fabs(self.cn), axis=1)]

    def __len__(self):
        return self.cn.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.spec[idx], self.cn[idx]
        if self.transform is not None:
            sample = [self.transform(x) for x in sample]
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return torch.Tensor(sample)


def get_dataloaders(csv_fn, batch_size, train_val_test_ratios=(0.7, 0.15, 0.15),
                                   sampling_exponent=0.6, n_coord_num=3):
    transform_list = transforms.Compose([ToTensor()])
    ds_train,  ds_val, ds_test = [CoordNumSpectraDataset(
            csv_fn, p, train_val_test_ratios, sampling_exponent, n_coord_num=n_coord_num, transform=transform_list)
        for p in ["train", "val", "test"]]

    train_sampler = WeightedRandomSampler(ds_train.sampling_weights_per_sample, replacement=True,
                                          num_samples=math.ceil(len(ds_train)/batch_size)*batch_size)
    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=0, 
                              pin_memory=False)
    val_loader = DataLoader(ds_val, 
                            batch_size=batch_size, 
                            num_workers=0, 
                            pin_memory=False)
    test_loader = DataLoader(ds_test, 
                            batch_size=batch_size, 
                            num_workers=0, 
                            pin_memory=False)

    return train_loader, val_loader, test_loader
