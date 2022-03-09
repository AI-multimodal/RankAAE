import math
import os, glob
import re
import itertools

import torch
import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

import matplotlib as mpl
from matplotlib.colors import colorConverter
from matplotlib import ticker as ticker
import seaborn as sns
import plotly.express as px


def get_style_correlations(ds, encoder):
    encoder.eval()
    val_spec = torch.tensor(ds.spec, dtype=torch.float32)
    styles = encoder(val_spec).clone().detach().cpu().numpy()
    nstyles = styles.shape[1]
    inter_style_sm = max([math.fabs(spearmanr(*styles[:, pair].T).correlation) \
                          for pair in itertools.combinations(range(nstyles), 2)])
   
    return inter_style_sm


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


def plot_spectra_variation(decoder, istyle, x=None, ax=None, n_spec=50, n_sampling=1000, amplitude=2):
    """
    Spectra variation plot by varying one of the styles.
    """
    decoder.eval()
    if n_sampling == None:
        c = np.linspace(*[-amplitude, amplitude], n_spec)
        c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False)
        spec_out = decoder(con_c).reshape(n_spec, -1).clone().cpu().detach().numpy()
        colors = sns.color_palette("hsv", n_spec)
    else:
        con_c = torch.randn([n_spec, n_sampling, decoder.nstyle])
        style_variation = torch.linspace(-amplitude, amplitude, n_spec)
        con_c[..., istyle] = style_variation[:,np.newaxis]
        con_c = con_c.reshape(n_spec * n_sampling, decoder.nstyle)
        spec_out = decoder(con_c).reshape(n_spec, n_sampling, 256).mean(axis=1).cpu().detach().numpy()
        colors = create_plotly_colormap(n_spec)
    for spec, color in zip(spec_out, colors):
        if x is None:
            ax.plot(spec, lw=0.8, c=color)
        else: 
            ax.plot(x, spec, lw=0.8, c=color)
    ax.set_title(f"Varying Style #{istyle+1}", y=1)


def find_top_models(model_path, test_ds, n=5):
    '''
    Find top 5 models with leas correlatioin among styles, in descending order of goodness.
    '''
    model_files = os.path.join(model_path, "job_*/final.pt")
    fn_list = sorted(glob.glob(model_files), 
                    key=lambda fn: int(re.search(r"job_(?P<num>\d+)/", fn).group('num')))
    style_cor_list = []
    model_list = []
    for fn in fn_list:
        model = torch.load(fn, map_location=torch.device('cpu'))
        style_cor_list.append(get_style_correlations(test_ds, model["Encoder"]))
        model_list.append(model)
    
    # get the indices for top n least correlated styles, the first entry is the best model.
    top_indices = np.argsort(style_cor_list)[:n]
    
    return [model_list[i] for i in top_indices]


def get_confusion_matrix(cn, style_cn, ax=None):
    """
    get donfusion matrix for a discrete descriptor, such as coordination number.
    """
    
    data_length = len(cn)
    thresh_grid = np.linspace(-3.5, 3.5, 700)
    cn_classes = (cn - 4).astype(int) # the minimum CN is 4 by default.
    cn_class_sets = list(set(cn_classes))

    cn4_f1_scores = [f1_score(style_cn < th, cn_classes<1,zero_division=0) for th in thresh_grid]
    cn6_f1_scores = [f1_score(style_cn > th, cn_classes>1,zero_division=0) for th in thresh_grid]
    cn45_thresh = thresh_grid[np.argmax(cn4_f1_scores)]
    cn56_thresh = thresh_grid[np.argmax(cn6_f1_scores)]

    # calculate confusion matrix
    sep_pred_cn_classes = (style_cn > cn45_thresh).astype('int') + (style_cn > cn56_thresh).astype('int')
    sep_confusion_matrix_ = confusion_matrix(cn_classes, sep_pred_cn_classes)
    if len(cn_class_sets)==1: # when only one class is available, special care is needed.
        cn = cn_class_sets[0].astype(int)
        sep_confusion_matrix = np.zeros((3,3),dtype=int)
        sep_confusion_matrix[cn, cn] = sep_confusion_matrix_[0,0]
    else:
        sep_confusion_matrix = sep_confusion_matrix_
    
    sep_threshold_f1_score = f1_score(cn_classes, sep_pred_cn_classes, average='weighted')

    result = {
        "F1 Score": np.round(sep_threshold_f1_score, 4).tolist(),
        "CN45 Threshold": np.round(cn45_thresh, 4).tolist(),
        "CN56 Threshold": np.round(cn56_thresh, 4).tolist()
    }

    if ax is not None:
        sns.set_palette('bright', 2)
        ax[0].plot(thresh_grid, cn4_f1_scores, label='CN4')
        ax[0].plot(thresh_grid, cn6_f1_scores, label='CN6')
        ax[0].axvline(cn45_thresh, c='blue')
        ax[0].axvline(cn56_thresh, c='orange')
        ax[0].legend(loc='lower left', fontsize=12)

        sns.heatmap(sep_confusion_matrix, cmap='Blues', annot=True, fmt='d', cbar=False, ax=ax[1],
                    xticklabels=[f'CN{cn+4}' for cn in range(3)],
                    yticklabels=[f'CN{cn+4}' for cn in range(3)])
        ax[1].set_title(f"F1 Score = {sep_threshold_f1_score:.1%}",fontsize=12)
        ax[1].set_xlabel("Pred")
        ax[1].set_ylabel("True")

        # color splitting plot
        cn_list = [4,5,6]
        colors = np.array(sns.color_palette("bright", len(cn_list)))
        test_colors = colors[cn_classes]
        test_colors = np.array([colorConverter.to_rgba(c, alpha=0.6) for c in test_colors])     

        random_style = np.random.uniform(style_cn.min(),style_cn.max(),data_length)
        ax[2].scatter(style_cn, random_style, s=10.0, color=test_colors, alpha=0.8)
        ax[2].set_xlabel("Style 2")
        ax[2].set_ylabel("Random")
        ax[2].set_xlim([style_cn.min()-1, style_cn.max()+1])
        ax[2].set_ylim([style_cn.min()-2, style_cn.max()+1])
        ax[2].axvline(cn45_thresh, c='gray')
        ax[2].axvline(cn56_thresh, c='gray')

        n = len(colors)
        axins = ax[2].inset_axes([0.02, 0.06, 0.5, 0.1])
        axins.imshow(np.arange(n).reshape(1,n), cmap=mpl.colors.ListedColormap(list(colors)),
                    interpolation="nearest", aspect="auto")
        axins.set_xticks(list(range(n)))
        axins.set_xticklabels([f"CN{i+4}" for i in range(n)])
        axins.tick_params(bottom=False, labelsize=10)
        axins.yaxis.set_major_locator(ticker.NullLocator())

    return result
    

def get_descriptor_style_relation(
    style, 
    descriptor, 
    ax = None,
    choice = ["R2", "Spearman"],
    fit = True
):
    """
    Calculate the relations between styles and descriptors including R^2, Spearman, Polynomial/Linear fitting etc.
    If axis is given, scatter plot of given descriptor and style is also plotted.
    """
    
    # Make sure "style" is sorted.
    sorted_index = np.argsort(style)
    style = style[sorted_index]
    descriptor = descriptor[sorted_index]

    # Initialize accuracy dictionary.
    accuracy = {
        "Spearman": None,
        "Linear": {
            "slope": None,
            "intercept": None,
            "R2": None
        },
        "Quadruple": {
            "Parameters": [None, None, None],
            "residue": None
        }
    }

    # R^2 for the linear fitting
    if "R2" in choice:
        result = stats.linregress(style, descriptor)
        accuracy["Linear"]["R2"] = round(float(result.rvalue**2), 4)
        accuracy["Linear"]["intercept"] = round(float(result.intercept), 4)
        accuracy["Linear"]["slope"] = round(float(result.slope), 4)
        fitted_value = result.intercept + style * result.slope

    # spearman coefficient
    if "Spearman" in choice:
        sm = spearmanr(style, descriptor).correlation
        accuracy["Spearman"] = round(float(sm), 4)
    
    if "Quadruple" in choice:
        p, info = Polynomial.fit(style, descriptor, 2, full=True)
        accuracy["Quadruple"]["Parameters"] = np.round(p.convert().coef, 4).tolist()
        accuracy["Quadruple"]["residue"] = np.round(info[0]/len(style), 4).tolist()
        fitted_value = p(style)

    if ax is not None:
        ax.scatter(style, descriptor, s=10.0, c='blue', edgecolors='none', alpha=0.8)
        if fit:
            ax.plot(style, fitted_value, lw=2, c='black', alpha=0.5)

    return accuracy


def model_evaluation(test_ds, model, return_reconstruct=True, return_accuracy=True):
    '''
    calculate reconstruction error for a given model, or accuracy.
    
    Returns:
    --------
    '''
    descriptors = test_ds.aux
    result = {
        "Style": {},
        "Input": None, 
        "Output": None,
        "Reconstruct Err": (None, None),
    }
    
    encoder = model['Encoder']
    decoder = model['Decoder']
    encoder.eval()
    
    # Get styles via encoder
    spec_in = torch.tensor(test_ds.spec, dtype=torch.float32)
    styles = encoder(spec_in).clone().detach()
    result["Input"] = spec_in

    if return_reconstruct:
        spec_out = decoder(styles).clone().cpu().detach().numpy()
        mae_list = []
        for s1, s2 in zip(spec_in, spec_out):
            mae_list.append(mean_absolute_error(s1, s2))
        result["Reconstruct Err"] = [
            round(np.mean(mae_list).tolist(),4),
            round(np.std(mae_list).tolist(),4)
        ]
        result["Output"] = spec_out

    if return_accuracy:
        styles = styles.numpy()
        for i in range(descriptors.shape[1]):
            if i==1: # CN
                result["Style"][i] = \
                    get_confusion_matrix(descriptors[:,i], styles[:,i], ax=None)
            else:
                result["Style"][i] = \
                    get_descriptor_style_relation(descriptors[:,i], styles[:,i], ax=None,
                                                  choice = ["R2", "Spearman", "Quadruple"])
    return result


def qqplot_normal(x, ax=None, grid=True):
    """
    Examine the "normality" of a distribution using qqplot.
    """
    data_length = len(x)
    
    # standardize input data, and calculate the z-score
    x_std = (x - x.mean())/x.std()
    z_score = sorted(x_std)
    
    # sample from standard normal distribution and calculate quantiles
    normal = np.random.randn(data_length)
    q_normal = np.quantile(normal, np.linspace(0,1,data_length))
    
    # make the q-q plot if ax is given
    if ax is not None:
        ax.plot(q_normal, z_score, ls='',marker='.', color='k')
        ax.plot([q_normal.min(),q_normal.max()],[q_normal.min(),q_normal.max()],
                 color='k',alpha=0.5)
        ax.grid(grid)
    return q_normal, z_score
