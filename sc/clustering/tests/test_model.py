from unittest import TestCase
from sc.clustering.model import EncodingBlock, DecodingBlock, Encoder, Decoder, GaussianSmoothing, DummyDualAAE
import torch
from torch import nn


class TestEncodingBlock(TestCase):
    def test_model(self):
        t = torch.ones((32, 1, 256))
        eb = EncodingBlock(1, 2, 256, 128, kernel_size=11, stride=2)
        self.assertEqual(eb(t).shape, (32, 2, 128))


class TestDecodingBlock(TestCase):
    def test_model(self):
        t = torch.ones((32, 14, 1))
        eb = DecodingBlock(14, 8, 1, excitation=1)
        self.assertEqual(eb(t).shape, (32, 8, 4))


class TestGaussianSmoothing(TestCase):
    def test_model(self):
        t = torch.rand((3, 256))
        sm = GaussianSmoothing(channels=1, kernel_size=17, sigma=3.0, dim=1)
        t = t.unsqueeze(dim=1)
        t = nn.functional.pad(t, (8, 8), mode='replicate')
        t = sm(t).squeeze(dim=1)
        self.assertEqual(t.shape, (3, 256))


class TestEncoder(TestCase):
    def test_model(self):
        t = torch.ones((64, 256))
        eb = Encoder()
        self.assertEqual(eb(t)[0].shape, (64, 2))
        self.assertEqual(eb(t)[1].shape, (64, 12))


class TestDecoder(TestCase):
    def test_model(self):
        tz, ty = torch.ones(32, 2), torch.ones(32, 12)
        eb = Decoder()
        self.assertEqual(eb(tz, ty).shape, (32, 256))


class TestDummyDualAAE(TestCase):
    def test_model(self):
        t = torch.ones((64, 256))
        eb = DummyDualAAE(use_cnn_dis=False)
        self.assertEqual(eb(t)[0].shape, (64, 256))

