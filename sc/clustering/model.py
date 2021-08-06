import math
import numbers
import torch
from torch import nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.neg() * ctx.alpha
        return grad_input, None


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_len, out_len, kernel_size=7, stride=2, excitation=4,
                 dropout_rate=0.2):
        super(EncodingBlock, self).__init__()
        if in_channels > 1:
            self.bn1 = nn.BatchNorm1d(in_channels, affine=False)
        else:
            self.bn1 = None
        self.relu1 = nn.PReLU(num_parameters=out_channels, init=0.01)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                               padding_mode='replicate', stride=in_len//(out_len*stride))
        self.bn2 = nn.BatchNorm1d(out_channels, affine=False)
        self.relu2 = nn.PReLU(num_parameters=out_channels, init=0.01)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                               stride=stride)

        if in_len > 10:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_1 = None
        self.fc1 = nn.Linear(in_len, excitation)
        self.relu_excit_1 = nn.PReLU(num_parameters=in_channels, init=0.01)
        self.fc2 = nn.Linear(excitation, out_len)
        self.relu_excit_2 = nn.PReLU(num_parameters=in_channels, init=0.01)
        if in_channels != out_channels:
            self.bn_excit = nn.BatchNorm1d(in_channels, affine=False)
            self.relu_excit_3 = nn.PReLU(num_parameters=out_channels, init=0.01)
            self.conv_excit = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                                        groups=math.gcd(in_channels, out_channels))
        else:
            self.bn_excit = None
            self.relu_excit_3 = None
            self.conv_excit = None

        if stride > 1 or (in_channels != out_channels):
            self.conv_short = nn.Conv1d(in_channels, out_channels, kernel_size=in_len//out_len, stride=in_len//out_len,
                                        groups=math.gcd(in_channels, out_channels))
            self.relu_short = nn.PReLU(num_parameters=out_channels, init=0.01)
        else:
            self.conv_short = None

    def forward(self, x):

        if self.bn1 is not None:
            out = self.bn1(x)
        else:
            out = x
        residual = out
        out = self.conv1(out)
        out = self.relu1(out)

        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu2(out)

        if self.conv_short is not None:
            res = self.conv_short(residual)
            res = self.relu_short(res)
        else:
            res = residual

        if self.dropout_1 is not None:
            excit = self.dropout_1(residual)
        else:
            excit = residual
        excit = self.fc1(excit)
        excit = self.relu_excit_1(excit)

        excit = self.fc2(excit)
        excit = self.relu_excit_2(excit)
        if self.conv_excit is not None:
            excit = self.bn_excit(excit)
            excit = self.conv_excit(excit)
            excit = self.relu_excit_3(excit)

        out = out + res + excit
        return out


class DecodingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, in_len, excitation=4, dropout_rate=0.2, out_len=None):
        super(DecodingBlock, self).__init__()
        if out_len is None:
            out_len = in_len * 4
        if in_len > 1:
            self.bn1 = nn.BatchNorm1d(in_channels, affine=False)
        else:
            self.bn1 = None
        self.relu1 = nn.PReLU(num_parameters=out_channels, init=0.01)
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(out_channels, affine=False)
        self.relu2 = nn.PReLU(num_parameters=out_channels, init=0.01)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=out_len//(in_len*2), stride=out_len//(in_len*2))

        if in_len > 10:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_1 = None
        self.fc1 = nn.Linear(in_len, excitation)
        self.relu_excit_1 = nn.PReLU(num_parameters=in_channels, init=0.01)
        self.fc2 = nn.Linear(excitation, out_len)
        self.relu_excit_2 = nn.PReLU(num_parameters=in_channels, init=0.01)
        if in_channels != out_channels:
            self.bn_excit = nn.BatchNorm1d(in_channels, affine=False)
            self.relu_excit_3 = nn.PReLU(num_parameters=out_channels, init=0.01)
            self.conv_excit = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1,
                                        groups=math.gcd(in_channels, out_channels))
        else:
            self.bn_excit = None
            self.relu_excit_3 = None
            self.conv_excit = None

        self.conv_short = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=out_len//in_len, stride=out_len//in_len,
                                             groups=math.gcd(in_channels, out_channels))
        self.relu_short = nn.PReLU(num_parameters=out_channels, init=0.01)

    def forward(self, x):
        if self.bn1 is not None:
            out = self.bn1(x)
        else:
            out = x
        residual = out
        out = self.conv1(out)
        out = self.relu1(out)

        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu2(out)

        res = self.conv_short(residual)
        res = self.relu_short(res)

        if self.dropout_1 is not None:
            excit = self.dropout_1(residual)
        else:
            excit = residual
        excit = self.fc1(excit)
        excit = self.relu_excit_1(excit)
        excit = self.fc2(excit)
        excit = self.relu_excit_2(excit)
        if self.conv_excit is not None:
            excit = self.bn_excit(excit)
            excit = self.conv_excit(excit)
            excit = self.relu_excit_3(excit)

        out = out + res + excit
        return out


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cpu'):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.to(device))
        self.groups = channels

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        if len(x.size()) - 2 == 1:
            conv = nn.functional.conv1d
        elif len(x.size()) - 2 == 2:
            conv = nn.functional.conv2d
        elif len(x.size()) - 2 == 3:
            conv = nn.functional.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(len(x.size()) - 2)
            )
        return conv(x, weight=self.weight, groups=self.groups)


class Encoder(nn.Module):
    """ front end part of discriminator and Q"""

    def __init__(self, dropout_rate=0.2, nstyle=2):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            EncodingBlock(in_channels=1, out_channels=4, in_len=256, out_len=128, kernel_size=11, stride=2,
                          excitation=4, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=128, out_len=64, kernel_size=11, stride=2, excitation=4,
                          dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=64, out_len=32, kernel_size=7, stride=2, excitation=2,
                          dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=32, out_len=16, kernel_size=7, stride=2, excitation=2,
                          dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=16, out_len=8, kernel_size=5, stride=2, excitation=1,
                          dropout_rate=dropout_rate)
        )
        self.lin3 = nn.Linear(32, nstyle)
        self.bn_style = nn.BatchNorm1d(nstyle, affine=False)

    def forward(self, spec):
        batch_size = spec.size()[0]
        output = spec.unsqueeze(dim=1)
        output = self.main(output)
        output = output.reshape(batch_size, 32)

        z_gauss = self.lin3(output)
        z_gauss = self.bn_style(z_gauss)

        return z_gauss


class CompactEncoder(nn.Module):
    """ front end part of discriminator and Q"""

    def __init__(self, dropout_rate=0.2, nstyle=2):
        super(CompactEncoder, self).__init__()
        self.main = nn.Sequential(
            EncodingBlock(in_channels=1, out_channels=4, in_len=256, out_len=64, kernel_size=11, stride=2,
                          excitation=4, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=64, out_len=16, kernel_size=7, stride=2, excitation=2,
                          dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=16, out_len=8, kernel_size=5, stride=2, excitation=1,
                          dropout_rate=dropout_rate)
        )
        self.lin3 = nn.Linear(32, nstyle)
        self.bn_style = nn.BatchNorm1d(nstyle, affine=False)

    def forward(self, spec):
        batch_size = spec.size()[0]
        output = spec.unsqueeze(dim=1)
        output = self.main(output)
        output = output.reshape(batch_size, 32)

        z_gauss = self.lin3(output)
        z_gauss = self.bn_style(z_gauss)

        return z_gauss


class Decoder(nn.Module):

    def __init__(self, dropout_rate=0.2, nstyle=2, debug=False, last_layer_activation='ReLu'):
        super(Decoder, self).__init__()

        if last_layer_activation == 'ReLu':
            ll_act = nn.ReLU()
        elif last_layer_activation == 'Softplus':
            ll_act = nn.Softplus(beta=2)
        else:
            raise ValueError(f"Unknow activation function \"{last_layer_activation}\", please use one available in Pytorch")

        self.main = nn.Sequential(
            DecodingBlock(in_channels=nstyle, out_channels=8, in_len=1, excitation=1,
                          dropout_rate=dropout_rate),
            DecodingBlock(in_channels=8, out_channels=4, in_len=4, excitation=2, dropout_rate=dropout_rate),
            DecodingBlock(in_channels=4, out_channels=4, in_len=16, excitation=2, dropout_rate=dropout_rate),
            DecodingBlock(in_channels=4, out_channels=4, in_len=64, excitation=4, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=2, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=2, out_channels=2, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=2, out_channels=2, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            nn.BatchNorm1d(2, affine=False),
            nn.Conv1d(2, 1, kernel_size=1, stride=1),
            ll_act
        )

        self.nstyle = nstyle
        self.debug = debug

    def forward(self, z_gauss):
        if self.debug:
            assert z_gauss.size()[1] == self.nstyle
        x = z_gauss.unsqueeze(dim=2)
        spec = self.main(x)
        spec = spec.squeeze(dim=1)
        return spec


class CompactDecoder(nn.Module):

    def __init__(self, dropout_rate=0.2, nstyle=2, debug=False, last_layer_activation='ReLu'):
        super(CompactDecoder, self).__init__()

        if last_layer_activation == 'ReLu':
            ll_act = nn.ReLU()
        elif last_layer_activation == 'Softplus':
            ll_act = nn.Softplus(beta=2)
        else:
            raise ValueError(f"Unknow activation function \"{last_layer_activation}\", please use one available in Pytorch")

        self.main = nn.Sequential(
            DecodingBlock(in_channels= nstyle, out_channels=8, in_len=1, excitation=1, out_len=8,
                          dropout_rate=dropout_rate),
            DecodingBlock(in_channels=8, out_channels=4, in_len=8, excitation=2, out_len=64, 
                          dropout_rate=dropout_rate),
            DecodingBlock(in_channels=4, out_channels=4, in_len=64, excitation=4, dropout_rate=dropout_rate),
            EncodingBlock(in_channels=4, out_channels=4, in_len=256, out_len=256, kernel_size=11, stride=1,
                          excitation=2, dropout_rate=dropout_rate),
            nn.BatchNorm1d(4, affine=False),
            nn.Conv1d(4, 1, kernel_size=1, stride=1),
            ll_act
        )

        self.nstyle = nstyle
        self.debug = debug

    def forward(self, z_gauss):
        if self.debug:
            assert z_gauss.size()[1] == self.nstyle
        x = z_gauss.unsqueeze(dim=2)
        spec = self.main(x)
        spec = spec.squeeze(dim=1)
        return spec


class DiscriminatorCNN(nn.Module):
    def __init__(self, hiden_size=64, channels=2, kernel_size=5, dropout_rate=0.2, nstyle=2, noise=0.1):
        super(DiscriminatorCNN, self).__init__()

        self.pre = nn.Sequential(
            nn.Linear(nstyle, hiden_size),
            nn.PReLU(num_parameters=hiden_size, init=0.01)
        )

        self.main = nn.Sequential(
            nn.BatchNorm1d(1, affine=False),
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, padding_mode='replicate'),
            nn.PReLU(num_parameters=channels, init=0.01),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=(kernel_size-1)//2, padding_mode='replicate'),
            nn.PReLU(num_parameters=channels, init=0.01),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      padding=(kernel_size-1)//2, padding_mode='replicate'),
            nn.PReLU(num_parameters=channels, init=0.01),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=(kernel_size-1)//2,
                      padding_mode='replicate'),
            nn.PReLU(num_parameters=channels, init=0.01),

            nn.BatchNorm1d(channels, affine=False),
            nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, padding_mode='replicate'),
            nn.PReLU(num_parameters=1, init=0.01)
        )

        self.post = nn.Sequential(
            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, 2),
            nn.LogSoftmax(dim=1)
        )

        self.nstyle = nstyle
        self.noise = noise

    def forward(self, x, alpha):
        if self.training:
            x = x + self.noise * torch.randn_like(x, requires_grad=False)
        x = ReverseLayerF.apply(x, alpha)
        x = self.pre(x)
        x = x.unsqueeze(dim=1)
        x = self.main(x)
        x = x.squeeze(dim=1)
        out = self.post(x)
        return out


class DiscriminatorFC(nn.Module):
    def __init__(self, hiden_size=50, dropout_rate=0.2, nstyle=2, noise=0.1):
        super(DiscriminatorFC, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nstyle, hiden_size),
            nn.PReLU(num_parameters=hiden_size, init=0.01),

            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, hiden_size),
            nn.PReLU(num_parameters=hiden_size, init=0.01),

            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, hiden_size),
            nn.PReLU(num_parameters=hiden_size, init=0.01),

            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, hiden_size),
            nn.PReLU(num_parameters=hiden_size, init=0.01),

            nn.BatchNorm1d(hiden_size, affine=False),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hiden_size, 2),
            nn.LogSoftmax(dim=1)
        )
        self.nstyle = nstyle
        self.noise = noise

    def forward(self, x, alpha):
        if self.training:
            x = x + self.noise * torch.randn_like(x, requires_grad=False)
        reverse_feature = ReverseLayerF.apply(x, alpha)
        out = self.main(reverse_feature)
        return out


class DummyDualAAE(nn.Module):
    def __init__(self, use_cnn_dis, cls_encoder, cls_decoder):
        super(DummyDualAAE, self).__init__()
        self.encoder = cls_encoder()
        self.decoder = cls_decoder()
        self.discriminator = DiscriminatorCNN() if use_cnn_dis else DiscriminatorFC()

    def forward(self, x):
        z = self.encoder(x)
        x2 = self.decoder(z)
        is_gau = self.discriminator(z, 0.3)
        return x2, is_gau
