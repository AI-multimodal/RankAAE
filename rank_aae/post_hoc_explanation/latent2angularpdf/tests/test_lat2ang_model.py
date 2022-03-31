from unittest import TestCase
from rank_aae.post_hoc_explanation.latent2angularpdf.lat2ang_model import Latent2AngularPDF
import torch

class TestLatent2AngularPDF(TestCase):
    def test_model(self):
        x = torch.ones((32, 5))
        l2a = Latent2AngularPDF()
        out = l2a(x)
        self.assertEqual(out.size(), (32, 64, 64))   