from typing import no_type_check_decorator
import torch
from torch import nn
import numpy as np
from sc.clustering.model import GaussianSmoothing


class TrainingLossGeneral():

    def __init__(
        self, 
        input = None, 
        max_epoch = None, 
        device=torch.device('cpu')
    ):
        self.max_epoch = max_epoch # the maximum epoch the loss function is calculated
        self.device = device
        self.input = input
    
    def __call__(self, *args, **kwargs):
        """
        Parameters
        ----------
        epoch : The current epoch.
        """
        
        raise NotImplementedError

class KendallConstraint(TrainingLossGeneral):
    def __init__(self, max_epoch=None, device=torch.device('cpu')):
        super.__init_(max_epoch=max_epoch, device=device)
    
    def __call__(self, epoch, input=None, model=None):
        pass
      
    
def kendall_constraint(descriptors, styles, activate=False, device=None):
    """
    Implement kendall_constraint. It runs on GPU.
    Kendall Rank Correlation Coefficeint:
        https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    
    Parameters
    ----------
    descriptors : array_like
        Array of hape (M, N) where M is the number of data points, N is the number of descriptors.
    styles : array_like
        It has the same shape of `descriptors`.

    Notes
    -----
    aux_target[i,j,k] = descriptors[i,k]-descriptors[j,k]
    
    """
    if device is None:
        device = torch.device('cpu')
    
    try:
        n_aux = styles.shape[1]
    except:
        raise

    aux_target = torch.sign(descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :])
    assert len(styles.size()) == 2
    aux_pred = styles[:, np.newaxis, :] - styles[np.newaxis, :, :]
    aux_len = aux_pred.size()[0]
    product = aux_pred * aux_target
    if activate:
        full_same_sel = product > 0
        full_opp_sel = product < 0
        aux_indices = torch.arange(n_aux, device=device)
        for i in range(n_aux):
            aux_sel = aux_indices == i
            n_same = max(torch.numel(product[full_same_sel & aux_sel]), 1)
            n_opp = max(torch.numel(product[full_opp_sel & aux_sel]), 1)
            product[full_same_sel & aux_sel] *= n_opp / max(n_same, n_opp)
    aux_loss = - product.sum() / ((aux_len**2 - aux_len) * n_aux)

    return aux_loss

def recon_loss(spec_in, spec_out, scale=False, mse_loss=None, device=None):
    """
    Reconstruction loss.

    Parameters
    ----------
    spec_in : array_like
        A 2-D array of a minibatch of spectra as the input to the encoder.
    spec_re : array_like
        A 2-D array of spectra as the output of decoder.
    """
    if device is None:
        device = torch.device('cpu')
    if mse_loss is None:
        mse_loss = nn.MSELoss().to(device)

    spec_in = spec_in.clone()

    if not scale:
        recon_loss = mse_loss(spec_out, spec_in)
    else:
        spec_scale = torch.abs(spec_out.mean(dim=1)) / torch.abs(spec_in.mean(dim=1))
        recon_loss = ((spec_scale - 1.0) ** 2).mean() * 0.1
        spec_scale = torch.clamp(spec_scale.detach(), min=0.7, max=1.3)
        recon_loss += mse_loss(spec_out,(spec_in.T * spec_scale).T)
    
    return recon_loss

def adversarial_loss(spec_in, styles, D, alpha, batch_size=100,  nll_loss=None, device=None):
    """
    Parameters
    ----------
    D : Discriminator
    """
    if device is None:
        device = torch.device('cpu')
    if nll_loss is None:
        nll_loss = nn.NLLLoss().to(device)

    nstyle = styles.size()[1]

    z_real_gauss = torch.randn(batch_size, nstyle, requires_grad=True, device=device)
    real_gauss_pred = D(z_real_gauss, alpha)
    real_gauss_label = torch.ones(batch_size, dtype=torch.long, requires_grad=False, device=device)
    
    fake_gauss_pred = D(styles, alpha)
    fake_guass_label = torch.zeros(spec_in.size()[0], dtype=torch.long, requires_grad=False,device=device)
            
    adversarial_loss = nll_loss(real_gauss_pred, real_gauss_label) \
                        + nll_loss(fake_gauss_pred, fake_guass_label)

    return adversarial_loss


def discriminator_loss(styles, D, batch_size=100,  loss_fn=None, device=None):
    """
    Parameters
    ----------
    D : Discriminator
    """
    if device is None:
        device = torch.device('cpu')
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss().to(device)

    z_real_gauss = torch.randn(batch_size, styles.size()[1], requires_grad=True, device=device)
    real_gauss_pred = D(z_real_gauss, None) # no gradient reversal, alpha=None
    real_gauss_label = torch.ones(batch_size, dtype=torch.long, requires_grad=False, device=device)
    
    fake_gauss_pred = D(styles, None)
    fake_guass_label = torch.zeros(styles.size()[0], dtype=torch.long, requires_grad=False,device=device)
            
    loss = loss_fn(real_gauss_pred, real_gauss_label) + loss_fn(fake_gauss_pred, fake_guass_label)

    return loss


def generator_loss(spec_in, encoder, D, loss_fn=None, device=None):
    if device is None:
        device = torch.device('cpu')
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss.to(device)

    styles = encoder(spec_in)
    fake_gauss_pred = D(styles, None) # no gradient reversal, alpha=None

    fake_guass_label = torch.zeros(styles.size()[0], dtype=torch.long, requires_grad=False,device=device)
            
    loss = loss_fn(fake_gauss_pred, fake_guass_label)

    return loss


def mutual_info_loss(spec_in, styles, encoder, decoder, mse_loss=None, device=None):
    """
    Sample latent space, reconstruct spectra and feed back to encoder to reconstruct latent space.
    Return the loss between the sampled and reconstructed latent spacc.
    """

    if device is None:
        device = torch.device('cpu')
    if mse_loss is None:
        mse_loss = nn.MSELoss().to(device)

    batch_size = spec_in.size()[0]
    nstyle = styles.size()[1]
    z_sample = torch.randn(batch_size, nstyle, requires_grad=False, device=device)
    
    z_recon = encoder(decoder(z_sample))
    mutual_info_loss = mse_loss(z_recon, z_sample)
    
    return mutual_info_loss

def smoothness_loss(spec_out, gs_kernel_size, mse_loss=None, device=None):
    """
    Return the smoothness loss.
    """
    if device is None:
        device = torch.device('cpu')
    if mse_loss is None:
        mse_loss = nn.MSELoss().to(device)

    gaussian_smoothing = GaussianSmoothing(
        channels=1, kernel_size=gs_kernel_size, sigma=3.0, dim=1,
        device = device
    )
    padding4smooth = nn.ReplicationPad1d(padding=(gs_kernel_size - 1)//2).to(device)
    spec_out_padded = padding4smooth(spec_out.unsqueeze(dim=1))
    spec_smoothed = gaussian_smoothing(spec_out_padded).squeeze(dim=1)
    smooth_loss_train = mse_loss(spec_out, spec_smoothed)

    return smooth_loss_train

def alpha(epoch_percentage, step=800, limit=0.7):
    """
    `epoch_percentage = epoch / max_epoch`
    """
    a = (2. / (1. + np.exp(-1.0E4 / step * epoch_percentage)) - 1) * limit
    return a
