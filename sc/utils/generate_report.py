import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import yaml
import sc.utils.analysis as analysis
from sc.clustering.dataloader import AuxSpectraDataset

def plot_report(test_ds, model, n_aux=5, title='report'):
    if n_aux == 5:
        name_list = ["CT", "CN", "OCN", "Rstd", "MOOD"]
    elif n_aux == 3:
        name_list = ["BVS", "CN", "OCN"]

    encoder = model['Encoder']
    decoder = model['Decoder']
    style_correlation = analysis.get_style_correlations(test_ds, encoder)
    
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32)
    test_grid = test_ds.grid
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
    ax7 = fig.add_subplot(gs[8:10,4:6])

    fig.suptitle(
        f"{title:s}\n"\
        f"Least correlation: {style_correlation:.4f}"
    )
    
    # Plot out synthetic spectra variation
    axs_spec = [ax1, ax2, axa, ax3, ax4, axb]
    for istyle, ax in enumerate(axs_spec):
        analysis.plot_spectra_variation(decoder, istyle, x=test_grid, n_spec=50, n_sampling=1000, ax=ax, amplitude=2)

    # Plot out descriptors vs styles
    styles_no_s2 = np.delete(test_styles,1, axis=1)
    descriptors_no_cn = np.delete(descriptors, 1, axis=1)
    name_list_no_cn = np.delete(name_list, 1, axis=0)
    for row in [4,5,6,7]:
        for col in [0,1,2,3]:
            ax = fig.add_subplot(gs[row,col])
            accuracy = analysis.get_descriptor_style_relation(
                styles_no_s2[:,col], 
                descriptors_no_cn[:,row-4], 
                ax=ax
            )
         
            ax.set_title(
                f"{name_list_no_cn[row-4]}: " +
                "{0:.2f}/{1:.2f}".format(accuracy["Linear"]["R2"], accuracy["Spearman"])
            )

    # Plot q-q plot of the style distribution
    for col in [0,1,2,3]:
        ax = fig.add_subplot(gs[8,col])
        _ = analysis.qqplot_normal(styles_no_s2[:,col], ax)
        if col > 0: col += 1 # skip style 2 which is CN
        ax.set_title(f'style_{col+1}')
    
    ax = fig.add_subplot(gs[9,3])
    _ = analysis.qqplot_normal(test_styles[:,1], ax)
    ax.set_title('style_2')

    # Plot out CN confusion matrix
    _ = analysis.get_confusion_matrix(descriptors[:,1].astype('int'), test_styles[:,1], [ax5, ax6, ax7])
    
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
        assert len(data_file_list) == 1, "Which data file are you going to use?"
        file_name = data_file_list[0]
    test_ds = AuxSpectraDataset(os.path.join(work_dir, file_name), split_portion="test", n_aux=5)
    
    #### Choose top n model based on inter style correlation ####
    model_path = os.path.join(work_dir, "training")
    top_models = analysis.find_top_models(model_path, test_ds, n=5)

    #### Generate report and calculate accuracy, reconstruction err,
    accuracy_n_model = {}
    for i, model in enumerate(top_models):
        result = analysis.model_evaluation(test_ds, model, return_reconstruct=True, return_accuracy=True)
        accuracy_n_model[i] = {
            'Accuracy': result["Accuracy"].round(4).tolist(),
            'Reconstruct_err': round(result["Reconstruct Err"][0].tolist(),4)
        }
        if i == 0: # Generate Report for best model
            fig = plot_report(test_ds, top_models[0],n_aux=5, title=args.output_name)
            spec_in = result["Input"]
            spec_out = result["Output"]
    average_accuracy = np.mean([v['Accuracy'] for v in accuracy_n_model.values()],axis=0)
    average_reconstruct_err = np.mean([v['Reconstruct_err'] for v in accuracy_n_model.values()])
    accuracy_n_model['Average'] = {
        'Accuracy': average_accuracy.round(4).tolist(),
        'Reconstruct_err': average_reconstruct_err.round(4).tolist()
    }

    #### Save report ####
    try:
        fig_path = os.path.join(work_dir, f"{args.output_name:s}"+".png")
        txt_path = os.path.join(work_dir, f"{args.output_name:s}"+".txt")
        spec_out_path = os.path.join(work_dir, f"{args.output_name:s}"+".out")
        spec_in_path = os.path.join(work_dir, f"{args.output_name:s}"+".in")
        fig.savefig(fig_path, bbox_inches='tight')
        yaml.dump(accuracy_n_model, open(txt_path, 'wt'))
        np.savetxt(spec_out_path,spec_out)
        np.savetxt(spec_in_path,spec_in)
        print("Success: training report saved!")
    except Exception as e:
        print(f"Fail: Cannot save training report: {e:s}")

if __name__ == "__main__":
    main()
