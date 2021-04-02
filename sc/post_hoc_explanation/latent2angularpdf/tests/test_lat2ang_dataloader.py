from unittest import TestCase
from sc.post_hoc_explanation.latent2angularpdf.lat2ang_dataloader import get_latent2apdf_dataloaders
import os

data_fn = os.path.join(os.path.dirname(__file__), "../../../data", "mini_latent2apdf_dataset.pkl")


class TestLatent2AngularPDFDataset(TestCase):
    def test_dataloader(self):
        dl_train, dl_val, dl_test = get_latent2apdf_dataloaders(data_fn, 512)
        self.assertEqual(len(dl_train), 32)
        self.assertEqual(len(dl_val), 32)
        self.assertEqual(len(dl_test.dataset), 32)

