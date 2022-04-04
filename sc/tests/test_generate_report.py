import sys
import torch
from sc.clustering.dataloader import AuxSpectraDataset
from sc.report.generate_report import *
import sc.report.analysis as analysis

class Test_GenerateReport():

    def test_reconstrusction_err(self):
        self.device = torch.device('cpu')

        data_dir = os.path.join(os.path.dirname(__file__), "data/")
        model_file = os.path.join(data_dir, 'training/job_1/final.pt')
        data_file = os.path.join(data_dir, 'feff_Fe_CT_CN_OCN_RSTD_MOOD_spec_202201171147_5000.csv')

        self.model = torch.load(model_file, map_location = self.device)
        self.ds = AuxSpectraDataset(data_file, split_portion = "test", n_aux = 5)

        result = analysis.evaluate_model(self.ds, self.model, device=self.device)
        mae, std = result["Reconstruct Err"]

        assert np.allclose(mae, 0.0316, atol=1e-4)
        assert np.allclose(std, 0.0095, atol=1e-4)
    
    def test_plot_spectra_variation(self):
        self.device = torch.device('cpu')
        data_dir = os.path.join(os.path.dirname(__file__), "data/")
        model_file = os.path.join(data_dir, 'training/job_1/final.pt')
        data_file = os.path.join(data_dir, 'feff_Fe_CT_CN_OCN_RSTD_MOOD_spec_202201171147_5000.csv')

        self.model = torch.load(model_file, map_location = self.device)
        self.ds = AuxSpectraDataset(data_file, split_portion = "val", n_aux = 5)
        spec_in = torch.tensor(self.ds.spec, dtype = torch.float32, device = self.device)
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

if __name__ == "__main__":
    main()
