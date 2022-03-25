import sys
import torch
from sc.clustering.dataloader import AuxSpectraDataset
from sc.utils.generate_report import *
import sc.utils.analysis as analysis

class Test_GenerateReport():

    def test_reconstrusction_err(self):
        # need to replace it with absolute path.
        # def setUp(self) -> None:
        # self.data_fn = os.path.join(os.path.dirname(__file__), "../../../data", "cu_feff_aux_bvs_cn_density.csv")

        
        model_file = 'tests/test_data/training/job_1/final.pt'
        data_file = 'tests/test_data/feff_Cu_CT_CN_OCN_RSTD_MOOD_spec_202201071543_4200.csv'
        self.model = torch.load(model_file, map_location=torch.device('cpu'))
        self.ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=5)

        reconstruction, accuracy = analysis.model_evaluation(self.ds, self.model)
        (mae, std), _ = reconstruction
        print(accuracy)
        assert np.allclose(mae, 0.0389, atol=1e-4)
        assert np.allclose(std, 0.0104, atol=1e-4)

if __name__ == "__main__":
    # model_file = 'tests/test_data/training/job_1/final.pt'
    # data_file = 'tests/test_data/feff_Cu_CT_CN_OCN_RSTD_MOOD_spec_202201071543_4200.csv'
    # model = torch.load(model_file, map_location=torch.device('cpu'))
    # ds = AuxSpectraDataset(data_file, split_portion="test", n_aux=5)

    
    # reconstruction, accuracy = model_evaluation(ds, model)
    # (mae, std), (spec_in, spec_out) = reconstruction
    # print(mae, std, accuracy)

    # import matplotlib.pyplot as plt
    # plt.plot(spec_in[1])
    # plt.plot(spec_out[1])
    # plt.savefig('tests/test_data/test.png')
    main()