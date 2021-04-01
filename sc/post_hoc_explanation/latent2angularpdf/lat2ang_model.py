from torch import nn
import torch

class Latent2AngularPDF(nn.Module):
    def __init__(self, lat_size=5, dropout_rate=0.05):
        super(Latent2AngularPDF, self).__init__()
        self.pre = nn.Sequential(nn.Linear(lat_size, 8),
                                 nn.ReLU(),
                                 nn.Dropout(p=dropout_rate))
        
        self.main = nn.Sequential(
            nn.BatchNorm2d(8, affine=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8, affine=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8, affine=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.BatchNorm2d(8, affine=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=9, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8, affine=False),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=17, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(8, affine=False),
            nn.Conv2d(in_channels=8, out_channels=2, kernel_size=11, groups=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(2, affine=False),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=6, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        out: torch.Tensor = self.pre(x)
        out = out.unsqueeze(dim=-1).unsqueeze(dim=-1)
        out = self.main(out)
        return out