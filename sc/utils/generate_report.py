from gettext import find
from turtle import st
from seaborn.rcmod import set_style
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sc.clustering.dataloader import AuxSpectraDataset
import math
import os, glob
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix, mean_absolute_error
import re
import itertools
import argparse
import plotly.express as px
from scipy.interpolate import interp1d
import yaml


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


def plot_spectra_variation(decoder, istyle, ax=None, n_spec=50, n_sampling=1000):
    decoder.eval()
    if n_sampling == None:
        c = np.linspace(*[-2, 2], n_spec)
        c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False)
        spec_out = decoder(con_c).reshape(n_spec, -1).clone().cpu().detach().numpy()
        colors = sns.color_palette("hsv", n_spec)
    else:
        con_c = torch.randn([n_spec, n_sampling, decoder.nstyle])
        style_variation = torch.linspace(-2, 2, n_spec)
        con_c[..., istyle] = style_variation[:,np.newaxis]
        con_c = con_c.reshape(n_spec * n_sampling, decoder.nstyle)
        spec_out = decoder(con_c).reshape(n_spec, n_sampling, 256).mean(axis=1).cpu().detach().numpy()
        colors = create_plotly_colormap(n_spec)
    for spec, color in zip(spec_out, colors):
            ax.plot(spec, lw=0.8, c=color)
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


def get_confusion_matrix(cn_classes, style, ax=None):
    
    thresh_grid = np.linspace(-3.5, 3.5, 700)
    cn_classes = cn_classes - 4 # the minimum CN is 4 by default.
    cn_class_sets = list(set(cn_classes))

    cn4_f1_scores = [f1_score(style < th, cn_classes<1,zero_division=0) for th in thresh_grid]
    cn6_f1_scores = [f1_score(style > th, cn_classes>1,zero_division=0) for th in thresh_grid]
    cn45_thresh = thresh_grid[np.argmax(cn4_f1_scores)]
    cn56_thresh = thresh_grid[np.argmax(cn6_f1_scores)]

    # calculate confusion matrix
    sep_pred_cn_classes = (style > cn45_thresh).astype('int') + (style > cn56_thresh).astype('int')
    sep_confusion_matrix_ = confusion_matrix(cn_classes, sep_pred_cn_classes)
    if len(cn_class_sets)==1: # when only one class is available, special care is needed.
        cn = cn_class_sets[0].astype(int)
        sep_confusion_matrix = np.zeros((3,3),dtype=int)
        sep_confusion_matrix[cn, cn] = sep_confusion_matrix_[0,0]
    else:
        sep_confusion_matrix = sep_confusion_matrix_
    
    sep_threshold_f1_score = f1_score(cn_classes, sep_pred_cn_classes, average='weighted')

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

    return sep_threshold_f1_score


def get_descriptor_accuracy(style, descriptor, ax=None):
    '''
    Scatter plot of given descriptor and style on the given axix, and returns the accuracy list of [R^2, spearman].
    '''
    _, _, r, _, _ = stats.linregress(style, descriptor)
    sm = spearmanr(style, descriptor).correlation
    accuracy = [round(float(r**2),4), round(float(sm),4)]
    if ax is not None:
        ax.scatter(style, descriptor, s=10.0, c='blue', edgecolors='none', alpha=0.8)
    return accuracy


def model_evaluation(test_ds, model, return_reconstruct=True, return_accuracy=True):
    '''
    calculate reconstruction error for a given model, or accuracy.
    
    Returns:
    --------
    '''
    descriptors = test_ds.aux
    accuracies = [None] * descriptors.shape[1]
    encoder = model['Encoder']
    decoder = model['Decoder']
    
    encoder.eval()
    spec_in = torch.tensor(test_ds.spec, dtype=torch.float32)
    styles = encoder(spec_in).clone().detach()

    reconstruct = [None, None]
    if return_reconstruct:
        spec_out = decoder(styles).clone().cpu().detach().numpy()
        mae_list = []
        for s1, s2 in zip(spec_in, spec_out):
            mae_list.append(mean_absolute_error(s1, s2))
        reconstruct = [(np.mean(mae_list), np.std(mae_list)), (spec_in, spec_out)]
    
    if return_accuracy:
        styles = styles.numpy()
        for i in range(len(accuracies)):
            if i==1: # CN
                accuracies[i] = get_confusion_matrix(descriptors[:,i], styles[:,i], ax=None)
            else:
                _, accuracies[i] = get_descriptor_accuracy(descriptors[:,i], styles[:,i], ax=None)
    
    return reconstruct, accuracies


def plot_report(test_ds, model, n_aux=5):
    if n_aux == 5:
        name_list = ["CT", "CN", "OCN", "Rstd", "MOOD"]
    elif n_aux == 3:
        name_list = ["BVs", "CN", "OCN"]

    encoder = model['Encoder']
    decoder = model['Decoder']
    style_correlation = get_style_correlations(test_ds, encoder)
    
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32)
    test_styles = encoder(test_spec).clone().detach().cpu().numpy()
    descriptors = test_ds.aux

    # generate a figure object to host all the plots
    fig = plt.figure(figsize=(12,24),constrained_layout=True)
    gs = fig.add_gridspec(12,6)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0:2,2:4])
    axa = fig.add_subplot(gs[0:2,4:6])
    ax3 = fig.add_subplot(gs[2:4,0:2])
    ax4 = fig.add_subplot(gs[2:4,2:4])
    axb = fig.add_subplot(gs[2:4,4:6])
    ax5 = fig.add_subplot(gs[4:6,4:6])
    ax6 = fig.add_subplot(gs[6:8,4:6])

    fig.suptitle(f"Least correlation: {style_correlation:.4f}")
    
    # Plot out synthetic spectra variation
    axs_spec = [ax1, ax2, axa, ax3, ax4, axb]
    for istyle, ax in enumerate(axs_spec):
        plot_spectra_variation(decoder, istyle, ax=ax, n_spec=50, n_sampling=1000)

    # Plot out descriptors vs styles
    styles_no_s2 = np.delete(test_styles,1, axis=1)
    descriptors_no_cn = np.delete(descriptors, 1, axis=1)
    name_list_no_cn = np.delete(name_list, 1, axis=0)
    for row in [4,5,6,7]:
        for col in [0,1,2,3]:
            ax = fig.add_subplot(gs[row,col])
            accuracy = get_descriptor_accuracy(styles_no_s2[:,col], 
                                               descriptors_no_cn[:,row-4], 
                                               ax=ax)
            ax.set_title(f"{name_list_no_cn[row-4]}: "+"{0:.2f}/{1:.2f}".format(*accuracy))

    # Plot out CN confusion matrix
    _ = get_confusion_matrix(descriptors[:,1].astype('int'), test_styles[:,1], [ax5, ax6])
    
    return fig



def main():
    #### Parse arguments ####
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_aux', type=int, default=3,
                        help="The number of auxiliary parameters")
    parser.add_argument('-m', '--n_free', type=int, default=1,
                        help="The number of free parameters")
    parser.add_argument('-w', '--work_dir', type=str, default='.',
                        help="The folder where the model and data are.")
    parser.add_argument('-f', '--data_file', type=str, default=None,
                        help="The name of the .csv data file.")
    parser.add_argument('-o', "--output_name", type=str, default="report",
                        help="The saved report figure.")
    parser.add_argument('-t', "--type", type=str, default="all",
                        help="The type of report: plot or accuracy file")

    args = parser.parse_args()
    work_dir = args.work_dir
    file_name = args.data_file
    
    #### Create test data set from file ####
    if file_name==None:  # if datafile name nor provided, search for it.
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1
        file_name = data_file_list[0]
    test_ds = AuxSpectraDataset(os.path.join(work_dir, file_name), split_portion="test", n_aux=5)
    
    #### Choose top n model based on inter style correlation ####
    model_path = os.path.join(work_dir, "training")
    top_models = find_top_models(model_path, test_ds, n=5)

    #### Generate report and calculate accuracy, reconstruction err,
    accuracy_n_model = {}
    for i, model in enumerate(top_models):
        if i == 0: # Generate Report for best model
            fig = plot_report(test_ds, top_models[0],n_aux=5)
        ((err, _),_), accuracies = model_evaluation(test_ds, model, return_reconstruct=True, return_accuracy=True)
        accuracy_n_model[i] = {
            'accuracy': accuracies,
            'reconstruct_err': err
        }
    accuracy_n_model['average'] = {
        'accuracy': np.mean([v['accuracy'] for v in accuracy_n_model.values()]),
        'reconstruct_err': np.mean([v['reconstruct_err'] for v in accuracy_n_model.values()])
    }
    #### Save report ####
    try:
        fig_path = os.path.join(work_dir, f"{args.output_name:s}"+".png")
        txt_path = os.path.join(work_dir, f"{args.output_name:s}"+".yml")
        fig.savefig(fig_path, bbox_inches='tight')
        yaml.dump(accuracy_n_model, open(txt_path, 'wt'))
        print("Success: training report saved!")
    except Exception as e:
        print(f"Fail: Cannot save training report: {e:s}")
    

if __name__ == "__main__":
    main()
