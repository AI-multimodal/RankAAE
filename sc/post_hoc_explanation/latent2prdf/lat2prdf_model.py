from torch import nn
import torch

class Latent2PRDF(nn.Module):
    def __init__(self, lat_size=5, dropout_rate=0.05):
        super(Latent2PRDF, self).__init__()
        
        self.main = nn.Sequential(
            nn.BatchNorm1d(lat_size, affine=False),
            nn.ConvTranspose1d(in_channels=lat_size, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(8, affine=False),
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(4, affine=False),
            nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(2, affine=False),
            nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=11, stride=2),
            nn.Softplus(beta=10)
        )

    def forward(self, x):
        out = x.unsqueeze(dim=-1)
        out = self.main(out)
        out = out.squeeze(dim=1)
        out = out[:, :-3]
        return out