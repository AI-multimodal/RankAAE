import sys
from sc.clustering.dataloader import AuxSpectraDataset
import torch

sys.path.append('/home/zliang/Documents/semi_clustering/')
from sc.utils.generate_report import *

class Test_GenerateReport():

    def test_reconstrusction_err(self):
        model_file = 'tests/test_data/training/job_1/final.pt'
        data_file = 'tests/test_data/feff_Cu_CT_CN_OCN_RSTD_MOOD_spec_202201071543_4200.csv'
        self.model = torch.load(model_file, map_location=torch.device('cpu'))
        self.ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=5)

        (mae, std), _ = reconstruction_err(self.ds, self.model)
        assert np.allclose(mae, 0.0389, atol=1e-4)
        assert np.allclose(std, 0.0104, atol=1e-4)

if __name__ == "__main__":
    model_file = 'tests/test_data/training/job_1/final.pt'
    data_file = 'tests/test_data/feff_Cu_CT_CN_OCN_RSTD_MOOD_spec_202201071543_4200.csv'
    model = torch.load(model_file, map_location=torch.device('cpu'))
    ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=5)

    (mae, std), (spec_in, spec_out) = reconstruction_err(ds, model)
    print(mae,std)
    import matplotlib.pyplot as plt
    plt.plot(spec_in[1])
    plt.plot(spec_out[1])
    plt.savefig('tests/test_data/test.png')