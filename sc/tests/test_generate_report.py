import os
import numpy as np
import torch
from sc.clustering.dataloader import AuxSpectraDataset
import sc.report.analysis as analysis
import sc.report.analysis_new as analysis_new

class Test_GenerateReport():

    device = torch.device('cpu')
    data_dir = os.path.join(os.path.dirname(__file__), "data/")
    data_file = os.path.join(data_dir, 'feff_V_CT_CN_OCN_RSTD_MOOD_spec_202209081430_7000.csv')
    model_file = os.path.join(data_dir, 'training/job_1/final.pt')
    model = torch.load(model_file, map_location = device)
    val_ds = AuxSpectraDataset(data_file, split_portion = "test", n_aux = 5)

    def test_reconstrusction_err(self):
        result = analysis.evaluate_model(self.val_ds, self.model, device=self.device)
        mae, std = result["Reconstruct Err"]
        assert np.allclose(mae, 0.0443, atol=1e-4)
        assert np.allclose(std, 0.0169, atol=1e-4)
    
    def test_plot_spectra_variation(self):
        spec_in = torch.tensor(self.val_ds.spec, dtype = torch.float32, device = self.device)
        self.styles = self.model["Encoder"](spec_in).clone().detach().cpu().numpy()
        analysis.plot_spectra_variation(
            self.model["Decoder"], 0, 
            n_spec = 50, 
            n_sampling = 1000, 
            true_range = True,
            styles = self.styles,
            amplitude = 2,
            device = self.device,
            ax = None,
            energy_grid = None
        )
    def test_reconstruct_evatuator_works(self):
        recon_evaluator = analysis_new.Reconstruct(name="recon")
        recon_evaluator.evaluate(self.val_ds, self.model, path_to_save=self.data_dir)
        pass

    def test_LossCurvePlotter_works(self):
        plotter = analysis_new.LossCurvePlotter()
        fig = plotter.plot_loss_curve(os.path.join(self.data_dir, "training/job_1/losses.csv"))
        fig.savefig(os.path.join(self.data_dir, "loss_curves.png"), bbox_inches="tight")
        print("Success: training report saved!")

if __name__ == "__main__":
    Test_GenerateReport().test_LossCurvePlotter_works()
