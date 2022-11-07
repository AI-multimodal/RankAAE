import math
import os
import itertools
import torch
import pickle
import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats
from scipy.stats import spearmanr, shapiro
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



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


def plot_spectra_variation(
    decoder, istyle,
    n_spec = 50, 
    n_sampling = 1000, 
    true_range = True,
    styles = None,
    amplitude = 2,
    device = torch.device("cpu"),
    ax = None,
    energy_grid = None,
    colors=None,
    plot_residual=False,
    z_range=None,
    **kwargs
):
    """
    Spectra variation plot by varying one of the styles.
    Parameters
    ----------
    istyle : int
        The column index of `styles` for which the variation is plotted.
    true_range : bool
        If True, sample from the 5th percentile to 95th percentile of a style, instead of 
        [-amplitude, +amplitude].
    n_sampling : int
        The number of random "styles" sampled. If zero, then set other styles to zero.
    amplitude : float
        The range from which styles are sampled. Effective only if `true_range` is False.
    style : array_like
        2-D array of complete styles. Effective and can't be None if `true_range` evaluates.
        True. The `istyle`th column 
    plot_residual : bool
        Weather to plot out the difference between two extrema instead of all variations.
    z_range: tuple(float, float)
        The range in which z varies to drive the spectral variation.
    """

    decoder.eval()
    if z_range is None:
        left, right = np.percentile(styles[:, istyle], [5, 95])
    else:
        left, right = z_range

    if n_sampling == 0: 
        c = np.linspace(left, right, n_spec)
        c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False, device=device)
        spec_out = decoder(con_c).reshape(n_spec, -1).clone().cpu().detach().numpy()
        style_variation = c
    else:
        # Create a 3-D array whose x,y,z dimensions correspond to: style variation, n_sampling, 
        # and number of styles. 
        con_c = torch.randn([n_spec, n_sampling, decoder.nstyle], device=device)
        assert len(styles.shape) == 2 # styles must be a 2-D array
        style_variation = torch.linspace(left, right, n_spec, device=device)
        # Assign the "layer" to be duplicates of `style_variation`
        con_c[..., istyle] = style_variation[:, np.newaxis]
        con_c = con_c.reshape(n_spec * n_sampling, decoder.nstyle)
        spec_out = decoder(con_c).reshape(n_spec, n_sampling, 256)
        # Average along the `n_sampling` dimsion.
        spec_out = spec_out.mean(axis = 1).cpu().detach().numpy()
    
    if ax is not None:
        if colors is None:
            colors = create_plotly_colormap(n_spec)
        assert len(colors) == n_spec
        for spec, color in zip(spec_out, colors):
            if energy_grid is None: # whether use energy as x-axis unit
                ax.plot(spec, c=color, **kwargs)
            elif plot_residual:  # whether plot raw variation or residual between extrema
                ax.plot(energy_grid, spec_out[-1]-spec_out[0], **kwargs)
                ax.set_ylim([-0.5, 0.5])
                break
            else:
                ax.plot(energy_grid, spec, c=color, **kwargs)
        ax.set_title(f"Style {istyle+1} varying from {left:.2f} to {right:.2f}", y=1)

    return style_variation, spec_out

def evaluate_all_models(
    model_path, test_ds, 
    device=torch.device('cpu')
):
    '''
    Sort models according to multi metrics, in descending order of goodness.
    '''

    # evaluate model
    result = {}
    for job in os.listdir(model_path):
        if job.startswith("job_"):
            model = torch.load(
                os.path.join(model_path, job, "final.pt"), 
                map_location = device
            )
            result[job] = evaluate_model(test_ds, model, device=device)
    
    return result

def load_evaluations(evaluation_path="./report_model_evaluations.pkl"):
    with open(evaluation_path, 'rb') as f:
        result = pickle.load(f)
    return result

def sort_all_models(
    result_dict, 
    sort_score = None, # sorting algorithm
    plot_score = False,
    ascending = True,
    top_n = None, 
    true_value = True, # whether annotate tru value or z score
    sorting_weight = None
):
    """
    Given the input result dict, calculate (and plot) the score matrix.
    Update the "rank" attribute and return the updated result dict.
    Add key "score" to the result.
    """
    # define the scores to be sorted
    score_names = [
        "Inter-style Corr", # 0
        "Reconstuction Err", # 1
        "Style_1 - CT Corr", # 2
        "Style_2 - CN Corr", # 3
        "Style_3 - OCN Corr", # 4
        "Style_4 - Rstd Corr", # 5
        "Style_5 - OO Corr", # 6
        
    ]
    scores = []
    jobs = []
    for job, result in result_dict.items():
        
        jobs.append(job)
        score = [
            result["Inter-style Corr"], # 0
            result["Reconstruct Err"][0], # 1
        ]
        for i in range(5):
            try:
                a = result["Style-descriptor Corr"][i]
                if i == 1:
                    score.append(a["F1 score"])
                else:
                    score.append(a["Spearman"])
            except KeyError:
                score.append(0)
            except TypeError:
                score.append(0)
        scores.append(score)

    jobs = np.array(jobs)
    scores = np.array(scores)
    # normalize the score so their color fall in the same range
    mu_std = np.stack((scores.mean(axis=0), scores.std(axis=0)), axis=1)
    z_scores = (scores - mu_std[:,0]) / mu_std[:,1]
    z_scores[:, (mu_std[:,1]==0)] = 0 # for columns of no variations, score=0

    
    # sort scroes
    if callable(sort_score):  # sort according to the `sort_score` algorithm 
        final_score = sort_score(z_scores, sorting_weight=sorting_weight)
    elif isinstance(sort_score, int) and sort_score>=0: # sort according to a single column
        final_score = scores[:, sort_score]
    else:
        final_score = np.arange(len(scores)) # no sorting
    
    rank = np.argsort(final_score)
    if (sort_score is not None) and (not ascending):
        rank = rank[::-1] # descending order
    
    ranked_scores = scores[rank]
    ranked_final_scores = final_score[rank]
    ranked_jobs = jobs[rank]
    ranked_z_scores = z_scores[rank]

    for i, (job, score) in enumerate(zip(ranked_jobs, ranked_final_scores)):
        result_dict[job]['Rank'] = i
        result_dict[job]['Score'] = round(float(score), 4)
        
    # plot out the heat map of scores
    fig = None
    if plot_score:
        if top_n is None or top_n > len(ranked_z_scores):
            top_n = len(ranked_z_scores)

        fig, ax = plt.subplots(figsize = (top_n, scores.shape[1]))
        ax.autoscale(enable=True)
        sns.heatmap(
            ranked_z_scores[:top_n].T,
            vmin = -3, vmax = 3,
            cmap = 'Blues', cbar = True, 
            annot = ranked_z_scores[:top_n].T if not true_value else ranked_scores[:top_n].T,
            ax = ax,
            yticklabels = [
                f"{name}\n{ms[0]:.3f}+-{ms[1]:.3f}" for name, ms in zip(score_names, mu_std)
            ],
            xticklabels = [
                f"{ranked_jobs[i]}: {ranked_final_scores[i]:.2f} "for i in range(top_n)
            ]
            
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='left',va='bottom')
        ax.tick_params(labelbottom=False,labeltop=True, axis='both', length=0,labelsize=15)

    return result_dict, ranked_jobs, fig


def get_confusion_matrix(true_labels, pred_styles, ax=None):
    """
    get donfusion matrix for a discrete descriptor, such as coordination number.
    """
    result = {
        "F1 score": None,
        "Thresholds": [],
    }
    
    data_length = len(true_labels)
    thresh_grid = np.linspace(-3.5, 3.5, 700)
    
    true_labels = np.array(true_labels).astype(int)
    true_label_sets = list(set(true_labels))
    pred_styles = np.array(pred_styles)

    true_classes = true_labels - min(true_labels)
    true_class_sets = list(set(true_classes))
    num_class = len(true_class_sets)

    f1_scores, max_f1_scores, thresholds = [], [], []
    for true_class in true_class_sets:
        f1_score_ = [f1_score(pred_styles<=th, true_classes<=true_class, zero_division=0) for th in thresh_grid]
        f1_scores.append(f1_score_)
        thresholds.append(thresh_grid[np.argmax(f1_score_)])
        max_f1_scores.append(np.max(f1_score_))
    
    pred_classes = np.zeros_like(pred_styles)
    for threshold in thresholds[:-1]:
        pred_classes += (pred_styles > threshold).astype('int')
    pred_labels = pred_classes + min(true_labels)

    # calculate confusion matrix
    sep_confusion_matrix_ = confusion_matrix(true_labels, pred_labels)
    if len(true_class_sets) > 10: # in case the input descriptor is noise
        return None
    elif len(true_class_sets)==1: # when only one class is available, special care is needed.
        true_labels = true_class_sets[0].astype(int)
        sep_confusion_matrix = np.zeros((num_class, num_class),dtype=int)
        sep_confusion_matrix[true_labels, true_labels] = sep_confusion_matrix_[0,0]
    else:
        sep_confusion_matrix = sep_confusion_matrix_
    
    average_f1_score = f1_score(true_labels, pred_labels, average='weighted')

    print(thresholds, max_f1_scores)
    result["F1 score"] = round(average_f1_score.tolist(), 4)
    result["Thresholds"] = [round(thresh, 4) for thresh in thresholds]

    if ax is not None:
        colors = np.array(sns.color_palette("bright", len(true_class_sets)))
        for i in range(len(f1_scores[:-1])):
            ax[0].plot(thresh_grid, f1_scores[i], label=true_label_sets[i], color=colors[i])
            ax[0].axvline(thresholds[i], color=colors[i])
        ax[0].legend(loc='lower left', fontsize=12)

        sns.heatmap(sep_confusion_matrix, cmap='Blues', annot=True, fmt='d', cbar=False, ax=ax[1],
                    xticklabels=[f'{label}' for label in true_label_sets],
                    yticklabels=[f'{label}' for label in true_label_sets])
        ax[1].set_title(f"F1 Score = {average_f1_score:.1%}",fontsize=12)
        ax[1].set_xlabel("Pred")
        ax[1].set_ylabel("True")

        # color splitting plot
        test_colors = colors[true_classes]
        test_colors = np.array([mpl.colors.colorConverter.to_rgba(c, alpha=0.8) for c in test_colors])
        random_styles = np.random.uniform(pred_styles.min(),pred_styles.max(),data_length)
        ax[2].scatter(pred_styles, random_styles, s=10.0, color=test_colors, alpha=0.8)
        ax[2].set_xlabel("Style 2")
        ax[2].set_ylabel("Random")
        ax[2].set_xlim([pred_styles.min()-1, pred_styles.max()+1])
        ax[2].set_ylim([pred_styles.min()-2, pred_styles.max()+1])
        for threshold in thresholds[:-1]:
            ax[2].axvline(threshold, c='gray')

        n = len(colors)
        axins = ax[2].inset_axes([0.02, 0.08, 0.5, 0.1])
        axins.imshow(np.arange(n).reshape(1,n), cmap=mpl.colors.ListedColormap(list(colors)),
                    interpolation="nearest", aspect="auto")
        axins.set_xticks(list(range(n)))
        axins.set_xticklabels([f"{label}" for label in true_label_sets])
        axins.tick_params(bottom=False, labelsize=10)
        axins.yaxis.set_major_locator(mpl.ticker.NullLocator())

    return result
    
def get_max_inter_style_correlation(styles):
    """
    The maximum of inter-style correlation.
    """
#    corr_list = [
#            math.fabs(spearmanr(*styles[:, pair].T).correlation) \
#                    for pair in itertools.combinations(range(styles.shape[1]), 2)
#    ]
    corr_list = [
        math.fabs(spearmanr(styles[:, i], styles[:, -1]).correlation)
        for i in range(styles.shape[1]-1)
    ]
    return round(max(corr_list), 4)
    

def get_descriptor_style_correlation(
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
    
    # Make sure "Style-descriptor Corr" is sorted.
    sorted_index = np.argsort(style)
    style = style[sorted_index]
    descriptor = descriptor[sorted_index]

    # mask out possible NaN values
    mask_nan = ~ (np.isnan(descriptor) | np.isnan(style))
    style = style[mask_nan]
    descriptor = descriptor[mask_nan]

    # Initialize accuracy dictionary.
    accuracy = {
        "Spearman": None,
        "Linear": {
            "slope": None,
            "intercept": None,
            "R2": None
        },
        "Quadratic": {
            "Parameters": [None, None, None],
            "residue": None,
            "R2": None
        }
    }

    # R^2 for the linear fitting
    if "R2" in choice:
        result = stats.linregress(style, descriptor)
        accuracy["Linear"]["R2"] = np.round(float(result.rvalue**2), 4).tolist()
        accuracy["Linear"]["intercept"] = np.round(float(result.intercept), 4).tolist()
        accuracy["Linear"]["slope"] = np.round(float(result.slope), 4).tolist()
        fitted_value = result.intercept + style * result.slope

    # spearman coefficient
    if "Spearman" in choice:
        sm = spearmanr(style, descriptor).correlation
        accuracy["Spearman"] = np.round(float(sm), 4).tolist()
    
    if "Quadratic" in choice:
        p, info = Polynomial.fit(style, descriptor, 2, full=True)
        accuracy["Quadratic"]["Parameters"] = np.round(p.convert().coef, 4).tolist()
        accuracy["Quadratic"]["residue"] = np.round(info[0]/len(style), 4).tolist()
        fitted_value = p(style)
        accuracy['Quadratic']["R2"] = \
            np.round(stats.linregress(fitted_value, descriptor).rvalue**2, 4).tolist()

    if ax is not None:
        ax.scatter(style, descriptor, s=10.0, c='blue', edgecolors='none', alpha=0.8)
        if fit:
            ax.plot(style, fitted_value, lw=2, c='black', alpha=0.5)

    return accuracy

    
def evaluate_model(
    test_ds, 
    model, 
    reconstruct = True, 
    accuracy = True,
    style = True,
    device = torch.device('cpu')
):
    '''
    calculate reconstruction error for a given model, or accuracy.
    
    Returns:
    --------
    '''
    descriptors = test_ds.aux
    result = {
        "Style-descriptor Corr": {},
        "Input": None, 
        "Output": None,
        "Reconstruct Err": (None, None),
        "Inter-style Corr": None  # Inter-style correlation
    }
    
    encoder = model['Encoder']
    decoder = model['Decoder']
    encoder.eval()
    
    # Get styles via encoder
    spec_in = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
    styles = encoder(spec_in)
    result["Input"] = spec_in.cpu().numpy()

    if reconstruct:
        spec_out = decoder(styles).clone().detach().cpu().numpy()
        mae_list = []
        for s1, s2 in zip(spec_in.cpu().numpy(), spec_out):
            mae_list.append(mean_absolute_error(s1, s2))
        result["Reconstruct Err"] = [
            round(np.mean(mae_list).tolist(),4),
            round(np.std(mae_list).tolist(),4)
        ]
        result["Output"] = spec_out

    if accuracy:
        styles = styles.clone().detach().cpu().numpy()
        for i in range(descriptors.shape[1]):
            if i==1: # CN
                result["Style-descriptor Corr"][i] = \
                    get_confusion_matrix(descriptors[:,i], styles[:,i], ax=None)
            else:
                result["Style-descriptor Corr"][i] = \
                    get_descriptor_style_correlation(descriptors[:,i], styles[:,i], ax=None,
                                                  choice = ["R2", "Spearman", "Quadratic"])
    if style:
        result["Inter-style Corr"] = get_max_inter_style_correlation(styles)

    return result


def qqplot_normal(x, ax=None, grid=True):
    """
    Examine the "normality" of a distribution using qqplot.
    Return the Shapiro statistic that represent the similarity of `x` to normality.
    """
    data_length = len(x)
    
    # standardize input data, and calculate the z-score
    x_std = (x - x.mean())/x.std()
    z_score = sorted(x_std)
    
    # sample from standard normal distribution and calculate quantiles
    normal = np.random.randn(data_length)
    q_normal = np.quantile(normal, np.linspace(0,1,data_length))

    # Calculate Shapiro statistic for z_score
    shapiro_statistic = shapiro(z_score).statistic
    # make the q-q plot if ax is given
    if ax is not None:
        ax.plot(q_normal, z_score, ls='',marker='.', color='k')
        ax.plot([q_normal.min(),q_normal.max()],[q_normal.min(),q_normal.max()],
                 color='k',alpha=0.5)
        ax.grid(grid)
    return shapiro_statistic
