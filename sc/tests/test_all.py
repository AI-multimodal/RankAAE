import os
import numpy as np
import torch
from sc.clustering.dataloader import AuxSpectraDataset
import sc.report.analysis as analysis
import sc.report.analysis_new as analysis_new
from sc.utils.parameter import Parameters
from sc.clustering.trainer import Trainer

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


class Test_Parameters():

    def test_from_yaml(self):

        fix_config_path = os.path.join(os.path.dirname(__file__), "./data/fix_config.yaml")
        p = Parameters.from_yaml(fix_config_path)
        
        assert p.ae_form == "FC"
        assert p.alpha_limit == 0.7172
    

    def test_update(self):
        
        parameter_dict = dict(
            nstyle = 2,
            weight_decay = 1e-2,
            lr_ratio_Reconn = 2.0, 
            optimizer_name = "AdamW",
            aux_weights = None, 
            kendall_activation = False
        )
        p = Parameters(parameter_dict)
        
        # test get method
        assert p.get('nstyle', 0) == 2
        assert p.get('nstyll', 0) == 0
        
        # test immutability
        try: 
            p.nstyle = 3
            assert False
        except TypeError:
            assert True

        # test update method as a tool to append new parameters
        p.update({"new_parameter1": 1.2, "new_parameter2": 1.3})
        assert p.new_parameter1 == 1.2
        assert p.new_parameter2 == 1.3

        # test update method as a tool to update existing attrbutes
        assert p.nstyle == 2
        assert p.kendall_activation == False
        parameter_dict.update({"nstyle": 3, "kendall_activation": True, "new_parameter3": 1.4})
        p.update(parameter_dict)
        assert p.nstyle == 3
        assert p.kendall_activation == True
        assert p.new_parameter3 == 1.4
    
class Test_Trainer():
    data_dir = os.path.join(os.path.dirname(__file__), "data/")
    data_file = os.path.join(data_dir, 'feff_V_CT_CN_OCN_RSTD_MOOD_spec_202209081430_7000.csv')
    fix_config_path = os.path.join(os.path.dirname(__file__), "./data/fix_config.yaml")
    p = Parameters.from_yaml(fix_config_path)
    
    def test_trainer_works(self):
        self.p.update({"max_epoch": 1, "trials": 1})
        trainer = Trainer.from_data(
            self.data_file,
            igpu = 0,
            verbose = False,
            work_dir = self.data_dir,
            config_parameters = self.p,
        )
        _ = trainer.train()


if __name__ == "__main__":
    Test_GenerateReport().test_LossCurvePlotter_works()
    Test_Parameters().test_update()
