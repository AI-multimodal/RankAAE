import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import yaml
import sc.utils.analysis as analysis
from sc.clustering.dataloader import AuxSpectraDataset

WEIGHT = [ 
    -5, # "Inter-style Corr"
    0, # "Reconstion Err"
    1, # "Style-Descriptor Corr 1"
    1, # "Style-Descriptor Corr 2"
    1, # "Style-Descriptor Corr 3"
    1, # "Style-Descriptor Corr 4"
    1, # "Style-Descriptor Corr 5"
]

def score(x, weight = [-5,0,1,1,1,1,1]):
    """
    columns of `x` respresents: 
        "Inter-style Corr", # 0
        "Reconstion Err", # 1
        "Style-Descriptor Corr 1", # 2
        "Style-Descriptor Corr 2", # 3
        "Style-Descriptor Corr 3", # 4
        "Style-Descriptor Corr 4", # 5
        "Style-Descriptor Corr 5" # 6
    """
    xx = x.copy()
    xx[:,0] = x[:,0] * weight[0]
    xx[:,1] = x[:,1] ** weight[1]
    xx[:,2] = x[:,2] * weight[2]
    xx[:,3] = x[:,3] * weight[3]
    xx[:,4] = x[:,4] * weight[4]
    xx[:,5] = x[:,5] * weight[5]

    return (xx[:,0] + np.sum(xx[:,2:])) / xx[:,1]


def plot_report(test_ds, model, n_aux=5, title='report'):
    if n_aux == 5:
        name_list = ["CT", "CN", "OCN", "Rstd", "MOOD"]
    elif n_aux == 3:
        name_list = ["BVS", "CN", "OCN"]

    encoder = model['Encoder']
    decoder = model['Decoder']
    result = analysis.evaluate_model(test_ds, model)
    style_correlation = result["Inter-style Corr"]
    
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
            
            # only correlated style has fitted line plotted.
            if col == row-4: 
                plot_fit = True
            else:
                plot_fit = False

            # for the first style (CT) use polynomial as fitting
            if col == 0:
                result_choice = ["R2", "Spearman", "Quadratic"]
            else:
                result_choice = ["R2", "Spearman"]
            
            accuracy = analysis.get_descriptor_style_relation(
                styles_no_s2[:,col], 
                descriptors_no_cn[:,row-4], 
                ax=ax,
                choice = result_choice,
                fit = plot_fit,
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
    

def save_evaluation_result(save_dir, file_name, model_results, save_spectra=False, top_n=5):
    """
    Input is a dictionary of result dictionaries of evaluate_model.
    And file name to save the resul.
    Information is saved to a txt file.
    """
    save_dict = {}
    for job, result in model_results.items():
        if result['Rank'] < top_n:
            save_dict[job] = {
                k: v for k, v in result.items() if k not in ["Input", "Output"]
            }
        if (result['Rank'] == 0) and save_spectra:
            spec_in = result["Input"]
            spec_out = result["Output"]

    yaml.dump(save_dict, open(os.path.join(save_dir, file_name+'.txt'), 'wt'))
    np.savetxt(os.path.join(save_dir, file_name+'.out'),spec_out)
    np.savetxt(os.path.join(save_dir, file_name+'.in'),spec_in)


def save_report_plot(save_dir, file_name, fig):
    fig.savefig(
        os.path.join(save_dir, file_name+"_best_model.png"),
        bbox_inches='tight'
    )

def save_model_selection_plot(save_dir, file_name, fig):
    fig.savefig(
        os.path.join(save_dir, file_name + "_model_selection.png"),
        bbox_inches = 'tight'
    )

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
    jobs_dir = os.path.join(work_dir, "training")
    file_name = args.data_file
    
    #### Create test data set from file ####
    if file_name==None:  # if datafile name nor provided, search for it.
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1, "Which data file are you going to use?"
        file_name = data_file_list[0]
    test_ds = AuxSpectraDataset(os.path.join(work_dir, file_name), split_portion="test", n_aux=5)
    
    #### Choose the 20 top model based on evaluation criteria ####
    model_results = analysis.evaluate_all_models(jobs_dir, test_ds) # models are not sorted
    model_results, sorted_jobs, fig_model_selection = analysis.sort_all_models( 
        model_results, 
        plot = True, 
        top_n = 20, 
        sort_score = lambda x: score(x, weight=WEIGHT)
    ) # models are sorted
    
    # genearte model selection scores plot
    if fig_model_selection is not None:
        save_model_selection_plot(work_dir, args.output_name, fig_model_selection)

    # generate report for top model
    top_model = torch.load(
            os.path.join(jobs_dir, sorted_jobs[0], "final.pt"), 
            map_location = torch.device('cpu')
    )
    fig_top_model = plot_report(test_ds, top_model, n_aux=5, title=args.output_name)
    save_report_plot(work_dir, args.output_name, fig_top_model)

    # save top 5 result 
    save_evaluation_result(work_dir, args.output_name, model_results, save_spectra=True, top_n=5)
    
    print("Success: training report saved!")

if __name__ == "__main__":
    main()
