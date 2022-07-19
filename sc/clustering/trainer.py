import shutil
import os
import logging
import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, spearmanr

import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sc.clustering.model import (
    DiscriminatorCNN, 
    DiscriminatorFC
)
from sc.clustering.dataloader import get_dataloaders
from sc.utils.parameter import AE_CLS_DICT, OPTIM_DICT, Parameters
from sc.utils.functions import (
    kendall_constraint, 
    recon_loss, 
    mutual_info_loss, 
    smoothness_loss, 
    discriminator_loss,
    generator_loss,
)


class Trainer:
    
    metric_weights = [1.0, -1.0, -0.01, -1.0, -1.0]
    gau_kernel_size = 17

    def __init__(
        self, 
        encoder, decoder, discriminator, device, train_loader, val_loader,
        verbose=True, work_dir='.', tb_logdir="runs", 
        config_parameters = Parameters({}), # initialize Parameters with an empty dictonary.
        logger = logging.getLogger("training"),
        loss_logger = logging.getLogger("losses")
    ):
        self.logger = logger
        self.loss_logger = loss_logger # for recording losses as a function of epochs
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.work_dir = work_dir
        self.tb_logdir = tb_logdir

        # update name space with config_parameters dictionary
        self.__dict__.update(config_parameters.to_dict())
        self.load_optimizers()
        self.load_schedulers()


    def train(self, callback=None):
        if self.verbose:
            para_info = torch.__config__.parallel_info()
            self.logger.info(para_info)

        # loss functions
        mse_loss = nn.MSELoss().to(self.device)
        CE_loss = nn.CrossEntropyLoss().to(self.device)

        # train network
        best_combined_metric = 10.0 # Initialize a guess for best combined metric.
        chkpt_dir = f"{self.work_dir}/checkpoints"
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir, exist_ok=True)
        best_chpt_file = None
        metrics = None
        
        # Record first line of loss values
        self.loss_logger.info( 
                "Epoch,Train_D,Val_D,Train_G,Val_G,Train_Aux,Val_Aux,Train_Recon,"
                "Val_Recon,Train_Smooth,Val_Smooth,Train_Mutual_Info,Val_Mutual_Info"
            )
        
        for epoch in range(self.max_epoch):
            # Set the networks in train mode (apply dropout when needed)
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()

            # Loop through the labeled and unlabeled dataset getting one batch of samples from each
            # The batch size has to be a divisor of the size of the dataset or it will return
            # invalid samples
            n_batch = len(self.loader)
            avg_mutual_info = 0.0
            for spec_in, aux_in in self.loader:
                spec_in = spec_in.to(self.device)
                if self.loader.dataset.aux is None:
                    aux_in = None
                else:
                    assert len(aux_in.size()) == 2
                    n_aux = aux_in.size()[-1]
                    aux_in = aux_in.to(self.device)
                
                spec_in += torch.randn_like(spec_in, requires_grad=False) * self.spec_noise
                styles = self.encoder(spec_in) # exclude the free style
                spec_out = self.decoder(styles) # reconstructed spectra

                # Kendall constraint
                self.zerograd()
                styles = self.encoder(spec_in)
                aux_loss_train = kendall_constraint(
                    aux_in, styles[:,:n_aux], 
                    activate=self.kendall_activation,
                    device=self.device
                )
                aux_loss_train.backward()
                self.optimizers["correlation"].step()

                # Init gradients, reconstruction loss
                self.zerograd()
                spec_out  = self.decoder(self.encoder(spec_in)) # retain the graph?
                recon_loss_train = recon_loss(
                    spec_in, spec_out, 
                    scale=self.use_flex_spec_target
                )
                recon_loss_train.backward()
                self.optimizers["reconstruction"].step()

                # Init gradients, mutual information loss
                self.zerograd()
                styles = self.encoder(spec_in)
                mutual_info_loss_train = mutual_info_loss(
                    spec_in, styles,
                    encoder=self.encoder, 
                    decoder=self.decoder, 
                    mse_loss=mse_loss, 
                    device=self.device
                )
                mutual_info_loss_train.backward()
                self.optimizers["mutual_info"].step()
                avg_mutual_info += mutual_info_loss_train.item()

                # Init gradients, smoothness loss
                if epoch < 500: # turn off smooth loss after 500
                    self.zerograd()
                    spec_out  = self.decoder(self.encoder(spec_in)) # retain the graph?
                    smooth_loss_train = smoothness_loss(
                        spec_out, 
                        gs_kernel_size=self.gau_kernel_size,
                        device=self.device
                    )
                    smooth_loss_train.backward()
                    self.optimizers["smoothness"].step()
                else:
                    smooth_loss_train = torch.tensor(0) 
                
                # Init gradients, discriminator loss
                self.zerograd()
                styles = self.encoder(spec_in)

                dis_loss_train = discriminator_loss(
                    styles, self.discriminator, 
                    batch_size=self.batch_size, 
                    loss_fn=CE_loss,
                    device=self.device
                )
                dis_loss_train.backward()
                self.optimizers["discriminator"].step()


                # Init gradients, generator loss
                self.zerograd()
                gen_loss_train = generator_loss(
                    spec_in, self.encoder, self.discriminator, 
                    loss_fn=CE_loss,
                    device=self.device
                )
                gen_loss_train.backward()
                self.optimizers["generator"].step()

                # Init gradients
                self.zerograd()

            ### Validation ###
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            
            spec_in_val, aux_in_val = [torch.cat(x, dim=0) for x in zip(*list(self.val_loader))]
            spec_in_val = spec_in_val.to(self.device)
            z = self.encoder(spec_in_val)
            spec_out_val = self.decoder(z)

            if self.loader.dataset.aux is None:
                aux_in = None
            else:
                assert len(aux_in_val.size()) == 2
                n_aux = aux_in_val.size()[-1]
                aux_in_val = aux_in_val.to(self.device)
            
            recon_loss_val = recon_loss(
                spec_in_val, 
                spec_out_val, 
                mse_loss=mse_loss, 
                device=self.device
            )
            aux_loss_val = kendall_constraint(
                aux_in_val, 
                z[:,:n_aux], 
                activate=self.kendall_activation
            )
            dis_loss_val = discriminator_loss(
                z, self.discriminator, 
                batch_size=len(z),
                loss_fn=CE_loss,
                device=self.device
            )
            gen_loss_val = generator_loss(
                spec_in_val, 
                self.encoder, 
                self.discriminator, 
                loss_fn=CE_loss, 
                device=self.device
            )
            smooth_loss_val = smoothness_loss(
                spec_out_val, 
                gs_kernel_size=self.gau_kernel_size,
                device=self.device
            )
            mutual_info_loss_val = mutual_info_loss(
                spec_in_val, z,
                encoder=self.encoder, 
                decoder=self.decoder, 
                mse_loss=mse_loss, 
                device=self.device
            )

            # Write losses to a file
            if epoch % 10 == 0:
                self.loss_logger.info(
                    f"{epoch:d},\t"
                    f"{dis_loss_train.item():.6f},\t{dis_loss_val.item():.6f},\t"
                    f"{gen_loss_train.item():.6f},\t{gen_loss_val.item():.6f},\t"
                    f"{aux_loss_train.item():.6f},\t{aux_loss_train.item():.6f},\t"
                    f"{recon_loss_train.item():.6f},\t{recon_loss_val.item():.6f},\t"
                    f"{smooth_loss_train.item():.6f},\t{smooth_loss_val.item():.6f},\t"
                    f"{mutual_info_loss_train.item():.6f},\t{mutual_info_loss_val.item():.6f},\t"
                )
            
            model_dict = {"Encoder": self.encoder,
                          "Decoder": self.decoder,
                          "Style Discriminator": self.discriminator}
            
            avg_mutual_info /= n_batch
            style_np = z.detach().clone().cpu().numpy().T
            style_shapiro = [shapiro(x).statistic for x in style_np]
            style_coupling = np.max(np.fabs(
                [
                    spearmanr(style_np[j1], style_np[j2]).correlation
                    for j1, j2 in itertools.combinations(range(style_np.shape[0]), 2)
                ]
            ))
            metrics = [min(style_shapiro), recon_loss_val.item(), avg_mutual_info, style_coupling,
                       aux_loss_val.item() if aux_in is not None else 0]
            
            combined_metric = - (np.array(self.metric_weights) * np.array(metrics)).sum()
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_chpt_file = f"{chkpt_dir}/epoch_{epoch:06d}_loss_{combined_metric:07.6g}.pt"
                torch.save(model_dict, best_chpt_file)

            for _, sch in self.schedulers.items():
                sch.step(combined_metric)

            if callback is not None:
                callback(epoch, metrics)
            
        # save the final model
        torch.save(model_dict, f'{self.work_dir}/final.pt')

        if best_chpt_file is not None:
            shutil.copy2(best_chpt_file, f'{self.work_dir}/best.pt')

        return metrics


    def zerograd(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discriminator.zero_grad()

    def get_style_distribution_plot(self, z):
        # noinspection PyTypeChecker
        fig, ax_list = plt.subplots(
            self.nstyle, 1, sharex=True, sharey=True, figsize=(9, 12))
        for istyle, ax in zip(range(self.nstyle), ax_list):
            sns.histplot(z[:, istyle], kde=False, color='blue', bins=np.arange(-3.0, 3.01, 0.2),
                         ax=ax, element="step")
        return fig


    def load_optimizers(self):
        opt_cls = OPTIM_DICT[self.optimizer_name]
        recon_optimizer = opt_cls(
            [
                {'params': self.encoder.parameters()}, 
                {'params': self.decoder.parameters()}
            ],
            lr = self.lr_ratio_Reconn * self.lr_base,
            weight_decay = self.weight_decay    
        )
        mutual_info_optimizer = opt_cls(
            [
                {'params': self.encoder.parameters()}, 
                {'params': self.decoder.parameters()}
            ],
            lr = self.lr_ratio_Mutual * self.lr_base
        )
        smooth_optimizer = opt_cls(
            [
                {'params': self.decoder.parameters()}
            ], 
            lr = self.lr_ratio_Smooth * self.lr_base,
            weight_decay = self.weight_decay
        )
        corr_optimizer = opt_cls(
            [
                {'params': self.encoder.parameters()}
            ],
            lr = self.lr_ratio_Corr * self.lr_base,
            weight_decay = self.weight_decay
        )
        dis_optimizer = opt_cls(
            [
                {'params': self.discriminator.parameters()}
            ],
            lr = self.lr_ratio_dis * self.lr_base,
            betas = (self.dis_beta * 0.9, self.dis_beta * 0.009 + 0.99)
        )

        gen_optimizer = opt_cls(
            [
                {'params': self.encoder.parameters()}
            ],
            lr = self.lr_ratio_gen * self.lr_base,
            betas = (self.gen_beta * 0.9, self.gen_beta * 0.009 + 0.99)
        )

        self.optimizers = {
            "reconstruction": recon_optimizer,
            "mutual_info": mutual_info_optimizer,
            "smoothness": smooth_optimizer,
            "correlation": corr_optimizer,
            "discriminator": dis_optimizer,
            "generator": gen_optimizer,
        }


    def load_schedulers(self):
        
        self.schedulers = {name:
            ReduceLROnPlateau(
                optimizer, mode="min", factor=self.sch_factor, patience=self.sch_patience, 
                cooldown=0, threshold=0.01,verbose=self.verbose
            ) 
            for name, optimizer in self.optimizers.items()
        }


    @classmethod
    def from_data(
        cls, csv_fn, 
        igpu=0, verbose=True, work_dir='.', 
        train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, 
        config_parameters = Parameters({}),
        logger = logging.getLogger("from_data"),
        loss_logger = logging.getLogger("losses")
    ):

        p = config_parameters
        assert p.ae_form in AE_CLS_DICT

        # load training and validation dataset
        dl_train, dl_val, _ = get_dataloaders(
            csv_fn, p.batch_size, (train_ratio, validation_ratio, test_ratio), n_aux=p.n_aux)


        # Use GPU if possible
        if torch.cuda.is_available():
            if verbose:
                logger.info("Use GPU")
            device = torch.device(f"cuda:{igpu}")
            for loader in [dl_train, dl_val]:
                loader.pin_memory = False
        else:
            if verbose:
                logger.warn("Use Slow CPU!")
            device = torch.device("cpu")

        # Load encoder, decoder and discriminator
        encoder = AE_CLS_DICT[p.ae_form]["encoder"](
            nstyle = p.nstyle, 
            dropout_rate = p.dropout_rate, 
            dim_in = p.dim_in, 
            n_layers = p.n_layers
        )
        decoder = AE_CLS_DICT[p.ae_form]["decoder"](
            nstyle = p.nstyle, 
            dropout_rate = p.dropout_rate, 
            last_layer_activation = p.decoder_activation, 
            dim_out = p.dim_out,
            n_layers = p.n_layers
        )
        if p.use_cnn_discriminator:
            discriminator = DiscriminatorCNN(
                nstyle=p.nstyle, dropout_rate=p.dis_dropout_rate, noise=p.dis_noise
            )
        else:
            discriminator = DiscriminatorFC(
                nstyle=p.nstyle, dropout_rate=p.dis_dropout_rate, noise=p.dis_noise,
                layers = p.FC_discriminator_layers
            )

        for net in [encoder, decoder, discriminator]:
            net.to(device)

        # Load trainer
        trainer = Trainer(
            encoder, decoder, discriminator, device, dl_train, dl_val,
            verbose=verbose, work_dir=work_dir,
            config_parameters=p, logger=logger, loss_logger=loss_logger
        )
        return trainer


 