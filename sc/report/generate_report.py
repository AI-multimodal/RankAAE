import torch
import numpy as np
import pickle
from collections import OrderedDict
from matplotlib import pyplot as plt
import os
import argparse
import json
import sc.report.analysis as analysis
import sc.report.analysis_new as analysis_new
from sc.utils.parameter import Parameters
from sc.clustering.dataloader import AuxSpectraDataset

def sorting_algorithm(x):
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

    weight = [-1, 0, 1, 1, 1, 1, 1]

    # if only weight[1] is non zero, turn on offset so the final score is non zero.
    off_set = 0 
    if np.sum(weight) == weight[1]:
        off_set = 1
    xx = x.copy()
    xx[:,0] = x[:,0] *  weight[0] # Inter-style Corr
    xx[:,1] = x[:,1] ** weight[1] # Reconstion Err
    xx[:,2] = x[:,2] *  weight[2] # Style1 - CT Corr
    xx[:,3] = x[:,3] *  weight[3] # Style2 - CN Corr
    xx[:,4] = x[:,4] *  weight[4] # Style3 - OCN Corr
    xx[:,5] = x[:,5] *  weight[5] # Style4 - Rstd Corr
    xx[:,6] = x[:,6] *  weight[6] # Style5 - MOOD Corr
    
    
    return (off_set + xx[:,0] + np.sum(xx[:,2:], axis=1)) / xx[:,1]


def plot_report(test_ds, model, n_aux=5, title='report', device = torch.device("cpu")):

    if n_aux == 5:
        name_list = ["CT", "CN", "OCN", "Rstd", "OO"]
    elif n_aux == 3:
        name_list = ["BVS", "CN", "OCN"]

    encoder = model['Encoder']
    decoder = model['Decoder']
    result = analysis.evaluate_model(test_ds, model, device=device)
    style_correlation = result["Inter-style Corr"]
    
    test_spec = torch.tensor(test_ds.spec, dtype=torch.float32, device=device)
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
        _ = analysis.plot_spectra_variation(
            decoder, istyle, 
            true_range = True,
            styles = test_styles,
            amplitude = 2,
            n_spec = 50, 
            n_sampling = 5000, 
            device = device,
            energy_grid = test_grid, 
            ax = ax
        )

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
            
            accuracy = analysis.get_descriptor_style_correlation(
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
        shapiro_statistic = analysis.qqplot_normal(styles_no_s2[:,col], ax)
        if col > 0: col += 1 # skip style 2 which is CN
        ax.set_title(f'style_{col+1}: {shapiro_statistic:.2f}')
    
    ax = fig.add_subplot(gs[9,3])
    shapiro_statistic = analysis.qqplot_normal(test_styles[:,1], ax)
    ax.set_title(f'style_2: {shapiro_statistic:.2f}')

    # Plot out CN confusion matrix
    _ = analysis.get_confusion_matrix(descriptors[:,1].astype('int'), test_styles[:,1], [ax5, ax6, ax7])
    
    return fig
    

def save_evaluation_result(save_dir, file_name, model_results, save_spectra=False, top_n=5):
    """
    Input is a dictionary of result dictionaries of evaluate_model.
    And file name to save the resul.
    Information is saved to a txt file.
    """
    save_dict = OrderedDict()
    sorted_top_n_jobs = list(range(top_n))
    for job, result in model_results.items():
        if result['Rank'] in sorted_top_n_jobs:
            sorted_top_n_jobs[result['Rank']] = job
    for job in sorted_top_n_jobs:
        result = model_results[job]
        save_dict[job] = {
            k: v for k, v in result.items() if k not in ["Input", "Output"]
        }
        if (result['Rank'] == 0) and save_spectra:
            spec_in = result["Input"]
            spec_out = result["Output"]
    with open(os.path.join(save_dir, file_name+'.json'), 'wt') as f:
        f.write(json.dumps(save_dict))
    np.savetxt(os.path.join(save_dir, file_name+'.out'),spec_out)
    np.savetxt(os.path.join(save_dir, file_name+'.in'),spec_in)


def save_model_evaluations(save_dir, file_name, result):
    with open(os.path.join(save_dir, file_name+"_model_evaluation.pkl"), "wb") as f:
        pickle.dump(result, f)


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
    parser.add_argument('-w', '--work_dir', type = str, default = '.',
                        help = "The folder where the model and data are.")  
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config for training parameter in YAML format')
    
    args = parser.parse_args()
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    config = Parameters.from_yaml(os.path.join(work_dir, args.config))
    

    jobs_dir = os.path.join(work_dir, "training")
    file_name = config.data_file
    
    device = torch.device("cpu") # device is cpu by default
    if config.gpu: 
        try:
            device = torch.device("cuda:0")
        except:
            device = torch.device("cpu")

    #### Create test data set from file ####
    if file_name == None:  # if datafile name nor provided, search for it.
        data_file_list = [f for f in os.listdir(work_dir) if f.endswith('.csv')]
        assert len(data_file_list) == 1, "Which data file are you going to use?"
        file_name = data_file_list[0]
    test_ds = AuxSpectraDataset(os.path.join(work_dir, file_name), split_portion = "val", n_aux = 5)
    
    #### Choose the 20 top model based on evaluation criteria ####
    model_results = analysis.evaluate_all_models(jobs_dir, test_ds, device=device) # models are not sorted

    model_results, sorted_jobs, fig_model_selection = analysis.sort_all_models( 
        model_results, 
        plot_score = True, 
        top_n = config.top_n, 
        sort_score = sorting_algorithm,
        ascending = False, # best model has the highest score
    ) # models are sorted
    save_model_evaluations(work_dir, config.output_name, model_results)
    
    # genearte model selection scores plot
    if fig_model_selection is not None:
        save_model_selection_plot(work_dir, config.output_name, fig_model_selection)

    # generate report for top model
    top_model = torch.load(
            os.path.join(jobs_dir, sorted_jobs[0], "final.pt"), 
            map_location = device
    )
    fig_top_model = plot_report(
        test_ds, 
        top_model, 
        n_aux = config.n_aux, 
        title = '-'.join([config.output_name, sorted_jobs[0]]), 
        device = device
    )

    save_report_plot(work_dir, config.output_name, fig_top_model)
    recon_evaluator = analysis_new.Reconstruct(name=config.output_name, device=device)
    recon_evaluator.evaluate(test_ds, top_model, path_to_save=work_dir)
    
    # save top 5 result 
    save_evaluation_result(work_dir, config.output_name, model_results, save_spectra=True, top_n=config.top_n)
    
    plotter = analysis_new.LossCurvePlotter()
    fig = plotter.plot_loss_curve(os.path.join(jobs_dir, sorted_jobs[0], "losses.csv"))
    fig.savefig("loss_curves.png", bbox_inches="tight")
    print("Success: training report saved!")

if __name__ == "__main__":
    main()
