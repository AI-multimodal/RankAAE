import math
import os
import itertools
import torch
import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from abc import ABC, abstractmethod

def create_plotly_colormap(n_colors):
    '''
    Xiaohui's implementation of getting spectra color map.
    '''
    plotly3_hex_strings = px.colors.sequential.Plotly3
    rgb_values = np.array([[int(f"0x{c_hex[i:i+2]}", 16) for i in range(1, 7, 2)] for c_hex in plotly3_hex_strings])
    x0 = np.linspace(1, n_colors, rgb_values.shape[0])
    x1 = np.linspace(1, n_colors, n_colors)
    target_rgb_values = np.stack([interp1d(x0, rgb_values[:, i], kind='cubic')(x1) for i in range(3)]).T.round().astype('int')
    target_rgb_strings = ["#"+"".join([f'{ch:02x}' for ch in rgb]) for rgb in target_rgb_values]
    return target_rgb_strings


class Reporter:
    """
    A class to evaluate training job (for all sub jobs).
    """
    def __init__(self):
        pass
    
    def collect_results(self):
        """
        Evaluate results from the models, if the results exist alrady, collect them.
        """
        pass
    def report(self):
        """
        Print (& plot) a report for all the model evaluations.
        """
        pass


class Evaluator(ABC):
    """
    A base class for evluating a specific property/score of a model.
    """
    def __init__(self, device=torch.device('cpu')):
        
        self.result = None
        self.device = device
    
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model.
        """
        pass

    @abstractmethod
    def plot(self, ax=None):
        """
        Plot out the result.
        """
        pass



class EvaluatorAll:
    """
    A base class to evaluate one model and produces a result dicionary.
    """
    def __init__(self, device=torch.device('cpu')):
        self.data = None,
        self.model = None,
        self.device = device,
    
    @classmethod
    def from_file(cls, data_path=None, model_path=None, device=torch.device('cpu')):
        """
        Initate an Evaluator instance from a model file and validation dataset.
        """
        eval = Evaluator(device=device)
        eval.load_data(data_path)
        eval.load_model(model_path)

        return eval
    
    def load_model(self, model_path=None):
        """
        Load model from a model file.
        """
        pass

    def load_data(self, data_path=None):
        """
        Load validation/testing data from a data path.
        """
        pass



class SpectraVariationEvaluator(Evaluator):
    
    def __init__(self, 
        n_spec=50, 
        n_sampling=1000, 
        amplitude = 2,
        device=torch.device('cpu')):
        
        super().__init__(device=device)
        self.n_spec = n_spec
        self.n_sampling = n_sampling
        self.amplitude = amplitude
        self.styles = None
        self.istyle = None
        self.model = None


    def evaluate(self, istyle, true_range = True):
        """
        Spectra variation plot by varying one of the styles.
        Parameters
        ----------
        istyle : int
            The column index of `styles` for which the variation is plotted.
        true_range : bool
            If True, sample from the 5th percentile to 95th percentile of a style, instead of 
            [-amplitude, +amplitude].
        amplitude : float
            The range from which styles are sampled. Effective only if `true_range` is False.
        style : array_like
            2-D array of complete styles. Effective and can't be None if `true_range` evaluates.
            True. The `istyle`th column 
        """
        decoder = self.model['Decoder']
        decoder.eval()
        
        if self.n_sampling == None: 
            c = np.linspace(*[-self.amplitude, self.amplitude], self.n_spec)
            c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
            con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False)
            spec_out = decoder(con_c).reshape(self.n_spec, -1).clone().cpu().detach().numpy()

        else:
            # Create a 3-D array whose x,y,z dimensions correspond to: style variation, n_sampling, 
            # and number of styles. 
            con_c = torch.randn([self.n_spec, self.n_sampling, decoder.nstyle], device = self.device)
            if true_range: # sample from the true range of a style 
                assert len(self.styles.shape) == 2 # styles must be a 2-D array
                style_range = np.percentile(self.styles[:, istyle], [5, 95])
                style_variation = torch.linspace(*style_range, self.n_spec, device = self.device)
            else: # sample from the absolute range
                style_variation = torch.linspace(-self.amplitude, self.amplitude, self.n_spec, device = self.device)
            # Assign the "layer" to be duplicates of `style_variation`
            con_c[..., istyle] = style_variation[:, np.newaxis]
            con_c = con_c.reshape(self.n_spec * self.n_sampling, decoder.nstyle)
            spec_out = decoder(con_c).reshape(self.n_spec, self.n_sampling, 256)
            # Average along the `n_sampling` dimsion.
            self.result = spec_out.mean(axis = 1).cpu().detach().numpy()
            self.istyle = istyle

    def plot(self, ax = None, energy_grid = None):
        
        assert self.istyle is not None, "Please evaluate first!"
        colors = create_plotly_colormap(self.n_spec)
        
        if ax is None: # create a standalone fig
            fig, ax = plt.subplot(figsize=(8,6))
        else:
            fig, ax = None, ax
        
        for spec, color in zip(self.result, colors):
            if energy_grid is None:
                ax.plot(spec, lw=0.8, c=color)
            else: 
                ax.plot(energy_grid, spec, lw=0.8, c=color)
        
        ax.set_title(f"Varying Style #{self.istyle+1}", y=1)
        
        return fig


