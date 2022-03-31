from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import numpy as np
from torchvision import transforms
from rank_aae.clustering.dataloader import ToTensor

class Latent2PRDFDataset(Dataset):
    def __init__(self, pkl_fn, set_name, element, transform=None):
        super(Latent2PRDFDataset, self).__init__()

        with open(pkl_fn, "rb") as f:
            ds_dict = pickle.load(f)

        self.mpid_iatom = ds_dict[set_name]['mpid_iatom']
        self.latent = ds_dict[set_name]['latent']
        self.prdf = ds_dict[set_name][f'PRDF-{element}']
        self.transform = transform
        assert len(self.mpid_iatom) == self.latent.shape[0]
        assert len(self.mpid_iatom) == self.prdf.shape[0]
        assert len(self.latent.shape) == 2
        assert len(self.prdf.shape) == 2

    def __len__(self):
        return self.latent.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.latent[idx], self.prdf[idx]
        if self.transform is not None:
            sample = [self.transform(x) for x in sample]
        return sample
        

def get_latent2prdf_dataloaders(pkl_fn, batch_size, element):
    transform_list = transforms.Compose([ToTensor()])
    ds_train,  ds_val, ds_test = [Latent2PRDFDataset(pkl_fn, set_name=p, element=element, transform=transform_list)
                                  for p in ["train", "val", "test"]]

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
