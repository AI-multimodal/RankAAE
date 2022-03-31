from unittest import TestCase
from rank_aae.post_hoc_explanation.latent2prdf.lat2prdf_model import Latent2PRDF
import torch

class TestLatent2PRDF(TestCase):
    def test_model(self):
        x = torch.ones((32, 5))
        l2a = Latent2PRDF()
        out = l2a(x)
        self.assertEqual(out.size(), (32, 100))