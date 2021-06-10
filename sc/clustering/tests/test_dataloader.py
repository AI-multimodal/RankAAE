from unittest import TestCase
from torchvision import transforms
from sc.clustering.dataloader import CoordNumSpectraDataset, ToTensor, get_dataloaders
import os




class TestCoordNumSpectraDataset(TestCase):

    def setUp(self) -> None:
        self.data_fn = data_fn = os.path.join(os.path.dirname(__file__), "../../../data", "ti_feff_cn_spec.csv")

    def test_dataset(self):
        transform_list = transforms.Compose([ToTensor()])
        dataset_feff_train = CoordNumSpectraDataset(self.data_fn, "train",
                                                    n_coord_num=3, transform=transform_list)
        dataset_feff_val = CoordNumSpectraDataset(self.data_fn, "val",
                                                  n_coord_num=3, transform=transform_list)
        dataset_feff_test = CoordNumSpectraDataset(self.data_fn, "test",
                                                   n_coord_num=3, transform=transform_list)

        self.assertEqual(len(dataset_feff_train), 3622)
        self.assertEqual(len(dataset_feff_val), 776)
        self.assertEqual(len(dataset_feff_test), 777)
        self.assertAlmostEqual(dataset_feff_test.sampling_weights_per_cn[0], 0.45, places=2)
        self.assertAlmostEqual(dataset_feff_test.sampling_weights_per_cn[1], 0.40, places=2)
        self.assertAlmostEqual(dataset_feff_test.sampling_weights_per_cn[2], 0.15, places=2)

    def test_dataloader(self):
        dl_train, dl_val, dl_test = get_dataloaders(self.data_fn, 512)
        self.assertEqual(len(dl_train), 8)
        self.assertEqual(len(dl_val), 2)
        self.assertEqual(len(dl_test.dataset), 777)
