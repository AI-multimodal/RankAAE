import os
from rank_aae.utils.parameter import Parameters

class Test_Parameters():

    def test_from_yaml(self):
        fix_config_path = os.path.join(os.path.dirname(__file__), "./test_data/fix_config.yaml")
        p = Parameters.from_yaml(fix_config_path)
        assert p.ae_form == "compact"
        assert p.alpha_limit == 1

if __name__ == "__main__":
    Test_Parameters().test_from_yaml()