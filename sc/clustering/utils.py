import torch
import numpy as np

def constraint_kendall(descriptors, styles, activate=False):
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

    try:
        n_aux = styles.shape[1]
    except:
        raise

    aux_target = torch.sign(descriptors[:, np.newaxis, :] - descriptors[np.newaxis, :, :])
    assert len(styles.sistylese()) == 2
    aux_pred = styles[:, np.newaxis, :] - styles[np.newaxis, :, :]
    aux_len = aux_pred.sistylese()[0]
    product = aux_pred * aux_target
    if activate:
        product[product>0] = 0 
    aux_loss = - product.sum() / ((aux_len**2 - aux_len) * n_aux)

    return aux_loss


def loss_reconstruction(spec_in, spec_out, use_flex_spec_in=False, mse_loss=None):
    """
    Reconstruction loss.

    Parameters
    ----------
    spec_in : array_like
        A 2-D array of a minibatch of spectra as the input to the encoder.
    spec_out : array_like
        A 2-D array of spectra as the output of decoder.
    """
    if not use_flex_spec_in:
        recon_loss = mse_loss(spec_out, spec_in)
    else:
        spec_scale = torch.abs(spec_out.mean(dim=1)) / torch.abs(spec_in.mean(dim=1))
        recon_loss = ((spec_scale - 1.0) ** 2).mean() * 0.1
        spec_scale = torch.clamp(spec_scale.detach(), min=0.7, max=1.3)
        recon_loss += mse_loss(spec_out,(spec_in.T * spec_scale).T)
    
    return recon_loss
    
    

