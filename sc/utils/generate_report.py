from seaborn.rcmod import set_style
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sc.clustering.dataloader import ToTensor, AuxSpectraDataset
import math
import os, glob
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix
import re
import itertools
import argparse
import plotly.express as px
from scipy.interpolate import interp1d


def get_style_correlations(val_ds, encoder, nstyles):
    encoder.eval()
    val_spec = torch.tensor(val_ds.spec, dtype=torch.float32)
    styles = encoder(val_spec).clone().detach().cpu().numpy()
    assert styles.shape[1] == nstyles
    inter_style_sm = max([math.fabs(spearmanr(*styles[:, pair].T).correlation) \
                          for pair in itertools.combinations(range(nstyles), 2)])
   
    return inter_style_sm

class ToTensor(object):
    def __call__(self, sample):
        return torch.Tensor(sample)

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


def plot_report(data_file, training_path, n_aux=3, n_free=1):

    if n_aux == 5:
        descriptor_list = ["CT", "CN", "OCN", "Rstd", "MOOD"]
    elif n_aux == 3:
        descriptor_list = ["BVs", "CN", "OCN"]

    # Read data and model
    
    val_ds = AuxSpectraDataset(data_file, split_portion="val", n_aux=n_aux)
    test_ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=n_aux)
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32)

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



    # Find the model in which styles are least correlated
    model_files = os.path.join(training_path, "job_*/final.pt")
    fn_list = sorted(glob.glob(model_files), 
                    key=lambda fn: int(re.search(r"job_(?P<num>\d+)/", fn).group('num')))
    nstyles = n_aux + n_free
    style_cor_list = []
    least_style_cor = 1 
    for fn in fn_list:
        final_model = torch.load(fn, map_location=torch.device('cpu'))
        style_cor = get_style_correlations(val_ds, final_model["Encoder"],nstyles)
        style_cor_list.append(style_cor)
        if style_cor < least_style_cor:
            least_style_cor = style_cor
            least_cor_model = final_model
            encoder = final_model["Encoder"]
            decoder = final_model["Decoder"]
    style_cor_list = np.array(style_cor_list)

    least_cor_job_index = np.argsort(style_cor_list)[0] + 1 # job number is one-based
    fig.suptitle(f"Least correlation of {least_style_cor:.4f} achieved in job_{least_cor_job_index}")


    # Plot out synthetic spectra variation
    axs_spec = [ax1, ax2, axa, ax3, ax4, axb]
    for istyle, ax in enumerate(axs_spec):
        plot_spectra_variation(decoder, istyle, ax=ax, n_spec=50, n_sampling=1000)


    # Plot out descriptors vs styles
    test_styles = encoder(test_spec).clone().detach().cpu().numpy()
    styles_no_s2 = np.delete(test_styles,1, axis=1)
    descriptors_no_cn = np.delete(test_ds.aux, 1, axis=1)
    descriptor_list_no_cn = np.delete(descriptor_list, 1, axis=0)
    for row in [4,5,6,7]:
        for col in [0,1,2,3]:
            ax = fig.add_subplot(gs[row,col])
            ax.scatter(styles_no_s2[:,col], descriptors_no_cn[:,row-4],
                       s=20.0, c='blue', edgecolors='none', alpha=0.8)
            _, _, r, _, _ = stats.linregress(styles_no_s2[:,col], descriptors_no_cn[:,row-4])
            sm = spearmanr(styles_no_s2[:,col], descriptors_no_cn[:,row-4]).correlation
            ax.set_title(f"{descriptor_list_no_cn[row-4]}: {r**2:.2f}/{sm:.2f}")


    # Plot out CN confusion matrix
    iclasses  = (test_ds.aux[:, 1]).astype('int')
    min_coord_num = iclasses.min()
    iclasses = iclasses - min_coord_num

    thresh_grid = np.linspace(-3.5, 1.5, 400)
    cn4_f1_scores = [f1_score(test_styles[:, 1] < th, iclasses<1) for th in thresh_grid]
    cn6_f1_scores = [f1_score(test_styles[:, 1] > th, iclasses>1) for th in thresh_grid]
    cn45_thresh = thresh_grid[np.argmax(cn4_f1_scores)]
    cn56_thresh = thresh_grid[np.argmax(cn6_f1_scores)]

    sep_pred_iclasses = (test_styles[:, 1] > cn45_thresh).astype('int') + (test_styles[:, 1] > cn56_thresh).astype('int')
    sep_confusion_matrix = confusion_matrix(iclasses, sep_pred_iclasses)
    sep_threshed_f1_score = f1_score(iclasses, sep_pred_iclasses, average='weighted')

    axs4 = [ax6, ax5]
    sns.set_palette('bright', 2)
    axs4[0].plot(thresh_grid, cn4_f1_scores, label='CN4')
    axs4[0].plot(thresh_grid, cn6_f1_scores, label='CN6')
    axs4[0].axvline(cn45_thresh, c='blue')
    axs4[0].axvline(cn56_thresh, c='orange')
    axs4[0].legend(loc='lower left', fontsize=12)

    sns.heatmap(sep_confusion_matrix, cmap='Blues', annot=True, fmt='d', cbar=False, ax=axs4[1],
                xticklabels=[f'CN{cn+4}' for cn in range(3)],
                yticklabels=[f'CN{cn+4}' for cn in range(3)])
    axs4[1].set_title(f"F1 Score = {sep_threshed_f1_score:.1%}",fontsize=12)
    axs4[1].set_xlabel("Pred")
    axs4[1].set_ylabel("True")

    return fig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_aux', type=int, default=3,
                        help="The number of auxiliary parameters")
    parser.add_argument('-m', '--n_free', type=int, default=1,
                        help="The number of free parameters")
    parser.add_argument('-w', '--work_dir', type=str, default='.',
                        help="The folder where the model and data are.")
    parser.add_argument('-f', '--data_file', type=str, default=None,
                        help="The name of the .csv data file.")
    parser.add_argument('-o', "--output_name", type=str, default="report.png",
                        help="The saved report figure.")
    
    args = parser.parse_args()
    work_dir = args.work_dir
    file_name = args.data_file

    # if datafile name nor provided, search for it.
    training_path = os.path.join(work_dir, "training")
    if file_name==None: 
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1
        file_name = data_file_list[0]
    file_path = os.path.join(work_dir, file_name)
    
    fig = plot_report(file_path, training_path, n_aux=5)

    # Save report
    try:
        fig_path = os.path.join(work_dir, f"{args.output_name:s}")
        fig.savefig(fig_path, bbox_inches='tight')
        print("Success: training report saved!")
    except Exception as e:
        print(f"Fail: Cannot save training report: {e:s}")

if __name__ == "__main__":
    main()
