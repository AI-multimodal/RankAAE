from collections import namedtuple


from rank_aae.clustering.model import (
    CompactDecoder, 
    CompactEncoder, 
    Encoder, 
    Decoder, 
    QvecDecoder, 
    QvecEncoder, 
    FCDecoder,
    FCEncoder,
)

AE_CLS_DICT = {
    "normal": {
        "encoder": Encoder, 
        "decoder": Decoder
    },
    "compact": {
        "encoder": CompactEncoder, 
        "decoder": CompactDecoder
    },
    "qved": {
        "encoder": QvecEncoder, 
        "decoder": QvecDecoder
    },
    "FC": {
        "encoder": FCEncoder, 
        "decoder": FCDecoder
    }
}

class parameters(namedtuple('Point', ['x', 'y'])):
    def __init__(self):
        


class Parameters():
    """
    A parameter object that maps all dictionary keys into its name space.
    """
    def __init__(self):
        
        """
        self, 
        encoder, decoder, discriminator, device, train_loader, val_loader,
        max_epoch=300, verbose=True, work_dir='.',
        tb_logdir="runs", base_lr=0.0001,
        """
        self.nstyle = 2,
        self.batch_size = 111, 
        self.grad_rev_beta = 1.1,  
        self.alpha_flat_step = 100, 
        self.alpha_limit = 2.0,
        self.sch_factor = 0.25, 
        self.sch_patience = 300, 
        self.spec_noise = 0.01, 
        self.weight_decay = 1e-2,
        self.lr_ratio_Reconn = 2.0, 
        self.lr_ratio_Mutual = 3.0, 
        self.lr_ratio_Smooth = 0.1,
        self.lr_ratio_Corr = 0.5, 
        self.lr_ratio_Style = 0.5, 
        self.optimizer_name = "AdamW",
        self.aux_weights = None, 
        self.kendall_activation = False,
        self.use_flex_spec_target = False
    
    def get(self, key, value):
        """
        Override the get method in the original dictionary parameters.
        """
        self.parameter_dict.get(key, value)
    
    def update(self, parameter_dict):
        self.__dict__.update(parameter_dict) # map keys to its name space. 

    @classmethod
    def from_yaml(cls, config_file_path):
        """
        Load parameter from a yaml file.
        """
        import yaml
        with open(config_file_path) as f:
            trainer_config = yaml.full_load(f)
        
        return Parameters(trainer_config)