from unittest import TestCase
from rank_aae.clustering.model import (
    CompactDecoder, 
    CompactEncoder, 
    EncodingBlock,
    DecodingBlock,
    Encoder, 
    Decoder, 
    QvecDecoder, 
    QvecEncoder, 
    GaussianSmoothing, 
    DummyDualAAE
)
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
        self.assertEqual(eb(t).shape, (64, 2))


class TestDecoder(TestCase):
    def test_model(self):
        tz = torch.ones(32, 2)
        eb = Decoder()
        self.assertEqual(eb(tz).shape, (32, 256))

class TestCompactDecoder(TestCase):
    def test_model(self):
        tz = torch.ones(32, 2)
        eb = CompactDecoder()
        self.assertEqual(eb(tz).shape, (32, 256))


class TestQvecDecoder(TestCase):
    def test_model(self):
        tz = torch.ones(32, 3)
        eb = QvecDecoder(nstyle=3)
        self.assertEqual(eb(tz).shape, (32, 12))


class TestCompactEncoder(TestCase):
    def test_model(self):
        t = torch.ones((64, 256))
        eb = CompactEncoder()
        self.assertEqual(eb(t).shape, (64, 2))


class TestQvecEncoder(TestCase):
    def test_model(self):
        t = torch.ones((64, 12))
        eb = QvecEncoder(nstyle=3)
        self.assertEqual(eb(t).shape, (64, 3))


class TestDummyDualAAE(TestCase):
    def test_model(self):
        t = torch.ones((64, 256))
        eb = DummyDualAAE(False, Encoder, Decoder)
        self.assertEqual(eb(t)[0].shape, (64, 256))
