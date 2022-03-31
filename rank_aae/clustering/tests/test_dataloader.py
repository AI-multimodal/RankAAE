from unittest import TestCase
from torchvision import transforms
from rank_aae.clustering.dataloader import AuxSpectraDataset, ToTensor, get_dataloaders
import os




class TestAuxSpectraDataset(TestCase):

    def setUp(self) -> None:
        self.data_fn = os.path.join(os.path.dirname(__file__), "../../../data", "cu_feff_aux_bvs_cn_density.csv")

    def test_dataset(self):
        transform_list = transforms.Compose([ToTensor()])
        dataset_feff_train = AuxSpectraDataset(self.data_fn, "train", transform=transform_list, n_aux=3)
        dataset_feff_val = AuxSpectraDataset(self.data_fn, "val", transform=transform_list, n_aux=3)
        dataset_feff_test = AuxSpectraDataset(self.data_fn, "test", transform=transform_list, n_aux=3)

        self.assertEqual(len(dataset_feff_train), 2983)
        self.assertEqual(len(dataset_feff_val), 639)
        self.assertEqual(len(dataset_feff_test), 640)

    def test_dataloader(self):
        dl_train, dl_val, dl_test = get_dataloaders(self.data_fn, 512, n_aux=3)
        self.assertEqual(len(dl_train), 6)
        self.assertEqual(len(dl_val), 2)
        self.assertEqual(len(dl_test.dataset), 640)
