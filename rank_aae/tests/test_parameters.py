import os
from rank_aae.utils.parameter import Parameters

class Test_Parameters():

    def test_from_yaml(self):
        fix_config_path = os.path.join(os.path.dirname(__file__), "./test_data/fix_config.yaml")
        p = ParametersIM.from_yaml(fix_config_path)
        assert p.ae_form == "compact"
        assert p.alpha_limit == 1
    
    def test_update(self):
        
        parameter_dict = dict(
            nstyle = 2,
            weight_decay = 1e-2,
            lr_ratio_Reconn = 2.0, 
            optimizer_name = "AdamW",
            aux_weights = None, 
            kendall_activation = False
        )

        p = ParametersIM(parameter_dict)
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
        parameter_dict.update({"nstyle": 3, "kendall_activation": True, "new_parameter3": 1.4})
        p.update(parameter_dict)
        assert p.nstyle == 3
        assert p.kendall_activation == True
        assert p.new_parameter3 == 1.4

if __name__ == "__main__":
    Test_Parameters().test_update()