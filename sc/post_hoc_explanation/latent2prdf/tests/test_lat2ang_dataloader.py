from unittest import TestCase
from sc.post_hoc_explanation.latent2prdf.lat2prdf_dataloader import get_latent2prdf_dataloaders
import os

data_fn = os.path.join(os.path.dirname(__file__), "../../../../data", "mini_latent2prdf_dataset.pkl")


class TestLatent2AngularPDFDataset(TestCase):
    def test_dataloader(self):
        dl_train, dl_val, dl_test = get_latent2prdf_dataloaders(data_fn, 4, 'Cu')
        self.assertEqual(len(dl_train), 8)
        self.assertEqual(len(dl_val), 8)
        self.assertEqual(len(dl_test), 8)
        self.assertEqual(len(dl_test.dataset), 32)

