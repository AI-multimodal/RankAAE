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
    

def plot_report(data_file,n_aux=3):

    # Read data and model
    
    val_ds = AuxSpectraDataset(data_file, split_portion="val", n_aux=n_aux)
    test_ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=n_aux)
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32)

    # generate a figure object to host all the plots
    fig = plt.figure(figsize=(8,16),constrained_layout=True)
    gs = fig.add_gridspec(8,4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0:2,2:4])
    ax3 = fig.add_subplot(gs[2:4,0:2])
    ax4 = fig.add_subplot(gs[2:4,2:4])
    ax5 = fig.add_subplot(gs[4:6,0:2])
    ax6 = fig.add_subplot(gs[4:6,2:4])
    ax7 = [fig.add_subplot(gs[6,i]) for i in [0,1,2,3]]
    ax8 = [fig.add_subplot(gs[7,i]) for i in [0,1,2,3]]



    # Find the model in which styles are least correlated
    fn_list = sorted(glob.glob("training/job_*/final.pt"), 
                    key=lambda fn: int(re.search(r"job_(?P<num>\d+)/", fn).group('num')))
    nstyles = 4
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
    decoder.eval()
    nspec_pc = 50
    c = np.linspace(*[-2, 2], nspec_pc)

    axs_spec = [ax1, ax2, ax3, ax4]
    for istyle in [0,1,2,3]:
        c2 = np.stack([np.zeros_like(c)] * istyle + [c] + [np.zeros_like(c)] * (decoder.nstyle - istyle - 1), axis=1)
        con_c = torch.tensor(c2, dtype=torch.float, requires_grad=False)
        spec_out = decoder(con_c).reshape(nspec_pc, -1).clone().cpu().detach().numpy()
        colors = sns.color_palette("hsv", nspec_pc)

        for i, (spec, color) in enumerate(zip(spec_out, colors)):
            axs_spec[istyle].plot(spec, lw=0.8, c=color)
            axs_spec[istyle].plot(spec, lw=0.8, c=color)
        title = f"Varying Style #{istyle+1}"
        axs_spec[istyle].set_title(title, y=1)


    # Plot out BVS vs styles
    test_styles = encoder(test_spec).clone().detach().cpu().numpy()

    bvs_test = test_ds.aux[:, 0]
    for i, ax in enumerate(ax7):
        ax.scatter(test_styles[:, i], bvs_test, s=20.0, c='blue', edgecolors='none', alpha=0.8)
        _,_, r, _, _ = stats.linregress(test_styles[:, i], bvs_test)
        sm = spearmanr(test_styles[:, i], bvs_test).correlation
        ax.set_title(f"BVS: {r**2:.2f}/{sm:.2f}")

    ocn_test = test_ds.aux[:, 2]
    for i, ax in enumerate(ax8):
        ax.scatter(test_styles[:, i], ocn_test, s=20.0, c='blue', edgecolors='none', alpha=0.8)
        _,_, r, _, _ = stats.linregress(test_styles[:, i], ocn_test)
        sm = spearmanr(test_styles[:, i], ocn_test).correlation


        ax.set_xlabel(f"Style {i+1:d}")
        ax.set_title(f"OCN: {r**2:.2f}/{sm:.2f}")


    # Plot out CN confusion matrix
    transform_list = transforms.Compose([ToTensor()])
    iclasses  = (test_ds.aux[:, 1]).astype('int')
    min_coord_num = iclasses.min()
    n_coord_num = iclasses.max() - min_coord_num + 1
    iclasses = iclasses - min_coord_num

    thresh_grid = np.linspace(-3.5, 1.5, 400)
    cn4_f1_scores = [f1_score(test_styles[:, 1] < th, iclasses<1) for th in thresh_grid]
    cn6_f1_scores = [f1_score(test_styles[:, 1] > th, iclasses>1) for th in thresh_grid]
    cn45_thresh = thresh_grid[np.argmax(cn4_f1_scores)]
    cn56_thresh = thresh_grid[np.argmax(cn6_f1_scores)]

    sep_pred_iclasses = (test_styles[:, 1] > cn45_thresh).astype('int') + (test_styles[:, 1] > cn56_thresh).astype('int')
    sep_confusion_matrix = confusion_matrix(iclasses, sep_pred_iclasses)
    sep_threshed_f1_score = f1_score(iclasses, sep_pred_iclasses, average='weighted')

    axs4 = [ax5, ax6]
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


    # Save report

    try:
        current_dir = os.getcwd().split('/')[-1]
        fig.savefig(f"report_{current_dir:s}.png", bbox_inches='tight')
        print("Success: training report saved!")
    except Exception as e:
        print(f"Fail: Cannot save training report: {e:s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_aux', type=int, default=3,
                        help="The number of auxiliary parameters")
    parser.add_arugment('-m', '--n_free', type=int, default=1,
                        help="The number of free parameters")
    parser.add_argument('-w', '--work_dir', type=str, default='.',
                        help="The folder where the model and data are.")
    parser.add_argument('-f', '--data_file', type=str, default=None,
                        help="The name of the .csv data file.")
    args = parser.parse_args()
    work_dir = args.work_dir
    file_name = args.data_file

    # if datafile name nor provided, search for it.
    if file_name==None: 
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1
        file_name = data_file_list[0]
    else:
        file_path = os.path.join(work_dir, args.data_file)
    
    plot_report(file_path, n_aux=5)


if __name__ == "__main__":
    main()
