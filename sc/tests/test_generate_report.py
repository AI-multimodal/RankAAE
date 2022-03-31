import sys
import torch
from sc.clustering.dataloader import AuxSpectraDataset
from sc.report.generate_report import *
import sc.report.analysis as analysis

class Test_GenerateReport():

    def test_reconstrusction_err(self):

        data_dir = os.path.join(os.path.dirname(__file__), "data/")
        model_file = os.path.join(data_dir, 'training/job_1/final.pt')
        data_file = os.path.join(data_dir, 'feff_Fe_CT_CN_OCN_RSTD_MOOD_spec_202201171147_5000.csv')

        self.model = torch.load(model_file, map_location=torch.device('cpu'))
        self.ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=5)

        result = analysis.evaluate_model(self.ds, self.model)
        mae, std = result["Reconstruct Err"]

        assert np.allclose(mae, 0.0316, atol=1e-4)
        assert np.allclose(std, 0.0095, atol=1e-4)

if __name__ == "__main__":
    main()
