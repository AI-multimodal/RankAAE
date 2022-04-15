import shutil
import os
import logging
import itertools
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, spearmanr

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sc.clustering.model import (
    GaussianSmoothing, 
    DummyDualAAE, 
    DiscriminatorCNN, 
    DiscriminatorFC
)
from sc.clustering.dataloader import get_dataloaders, AuxSpectraDataset, ToTensor
from sc.utils.parameter import AE_CLS_DICT, OPTIM_DICT, Parameters


class Trainer:
    
    metric_weights = [1.0, -1.0, -0.01, -1.0, -1.0]
    gau_kernel_size = 17

    def __init__(
        self, 
        encoder, decoder, discriminator, device, train_loader, val_loader,
        max_epoch=300, verbose=True, work_dir='.', tb_logdir="runs", 
        config_parameters = Parameters({}), # initialize Parameters with an empty dictonary.
        logger = logging.getLogger("training")
    ):
        self.logger = logger
        self.device = device
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.max_epoch = max_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.verbose = verbose
        self.work_dir = work_dir
        self.tb_logdir = tb_logdir

        # update name space with config_parameters dictionary
        self.__dict__.update(config_parameters.to_dict())

        self.gaussian_smoothing = GaussianSmoothing(
            channels=1, 
            kernel_size=self.gau_kernel_size, 
            sigma=3.0, dim=1,
            device = self.device).to(device)

        self.padding4smooth = nn.ReplicationPad1d(
            padding=(self.gau_kernel_size - 1) // 2).to(device)

        if self.verbose:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.work_dir, self.tb_logdir))
            self.tb_writer.add_graph(
                DummyDualAAE(
                    self.use_cnn_discriminator, 
                    self.encoder.__class__, 
                    self.decoder.__class__
                ), 
                iter(self.train_loader).next()[0] # example spec
            )

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

    def train(self, callback=None):
        if self.verbose:
            para_info = torch.__config__.parallel_info()
            self.logger.info(para_info)

        # loss functions
        mse_dis = nn.MSELoss().to(self.device)
        nll_loss = nn.NLLLoss().to(self.device)
        
        # optimizers
        opt_cls = OPTIM_DICT[self.optimizer_name]
        reconn_optimizer = opt_cls(
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
        adversarial_optimizer = opt_cls(
            [
                {'params': self.discriminator.parameters()},
                {'params': self.encoder.parameters()}
            ],
            lr = self.lr_ratio_Style * self.lr_base,
            betas = (self.grad_rev_beta * 0.9, 
            self.grad_rev_beta * 0.009 + 0.99)
        )

        schedulers = [
            ReduceLROnPlateau(
                optimizer, mode="min", factor=self.sch_factor, patience=self.sch_patience, 
                cooldown=0, threshold=0.01,verbose=self.verbose
            ) 
            for optimizer in [
                reconn_optimizer, mutual_info_optimizer, smooth_optimizer, corr_optimizer, 
                adversarial_optimizer
            ]
        ]

        # train network
        best_combined_metric = 10.0 # Initialize a guess for best combined metric.
        chkpt_dir = f"{self.work_dir}/checkpoints"
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir, exist_ok=True)
        best_chpt_file = None
        metrics = None
        for epoch in range(self.max_epoch):
            # Set the networks in train mode (apply dropout when needed)
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()

            # the weight of the gradient reversal
            alpha = (2. / (1. + np.exp(-1.0E4 / self.alpha_flat_step *
                                       epoch / self.max_epoch)) - 1) * self.alpha_limit

            # Loop through the labeled and unlabeled dataset getting one batch of samples from each
            # The batch size has to be a divisor of the size of the dataset or it will return
            # invalid samples
            n_batch = len(self.train_loader)
            avg_mutual_info = 0.0
            for spec_in, aux_in in self.train_loader:
                if self.train_loader.dataset.aux is None:
                    aux_in = None
                else:
                    assert len(aux_in.size()) == 2
                    n_aux = aux_in.size()[-1]
                    aux_in = aux_in.to(self.device)
                spec_in = spec_in.to(self.device)
                spec_target = spec_in.clone()
                spec_in = spec_in + \
                    torch.randn_like(
                        spec_in, requires_grad=False) * self.spec_noise

                # Init gradients, adversarial loss
                self.zerograd()
                z_real_gauss = torch.randn(
                    self.batch_size, self.nstyle, requires_grad=True, device=self.device)
                z_fake_gauss = self.encoder(spec_in)
                real_gauss_label = torch.ones(self.batch_size, dtype=torch.long, requires_grad=False,
                                              device=self.device)
                real_gauss_pred = self.discriminator(z_real_gauss, alpha)
                fake_guass_lable = torch.zeros(spec_in.size()[0], dtype=torch.long, requires_grad=False,
                                               device=self.device)
                fake_gauss_pred = self.discriminator(z_fake_gauss, alpha)
                adversarial_loss_train = nll_loss(real_gauss_pred, real_gauss_label) + \
                    nll_loss(fake_gauss_pred, fake_guass_lable)
                adversarial_loss_train.backward()
                adversarial_optimizer.step()

                # Correlation constration
                # Kendall Rank Correlation Coefficeint
                # https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
                if aux_in is not None:
                    self.zerograd()
                    z = self.encoder(spec_in)
                    aux_target = torch.sign(aux_in[:, np.newaxis, :] - aux_in[np.newaxis, :, :])
                    z_aux = z[:, :n_aux]
                    assert len(z_aux.size()) == 2
                    aux_pred = z_aux[:, np.newaxis, :] - z_aux[np.newaxis, :, :]
                    aux_len = aux_pred.size()[0]
                    aux_loss_train = - (aux_pred * aux_target).sum() / ((aux_len**2 - aux_len) * n_aux)
                    aux_loss_train.backward()
                    corr_optimizer.step()
                else:
                    aux_loss_train = None

                # Init gradients, reconstruction loss
                self.zerograd()
                spec_re = self.decoder(self.encoder(spec_in))
                if not self.use_flex_spec_target:
                    recon_loss_train = mse_dis(spec_re, spec_target)
                else:
                    spec_scale = torch.abs(spec_re.mean(
                        dim=1)) / torch.abs(spec_target.mean(dim=1))
                    recon_loss_train = ((spec_scale - 1.0) ** 2).mean() * 0.1
                    spec_scale = torch.clamp(spec_scale.detach(), min=0.7, max=1.3)
                    recon_loss_train += mse_dis(spec_re, (spec_target.T * spec_scale).T)
                recon_loss_train.backward()
                reconn_optimizer.step()

                # Init gradients, mutual information loss
                self.zerograd()
                z = torch.randn(self.batch_size, self.nstyle,
                                requires_grad=False, device=self.device)
                x_sample = self.decoder(z)
                z_recon = self.encoder(x_sample)
                mutual_info_loss_train = mse_dis(z_recon, z)
                mutual_info_loss_train.backward()
                mutual_info_optimizer.step()
                avg_mutual_info += mutual_info_loss_train.item()

                # Init gradients, smoothness loss
                self.zerograd()
                x_sample = self.decoder(z)
                x_sample_padded = self.padding4smooth(
                    x_sample.unsqueeze(dim=1))
                spec_smoothed = self.gaussian_smoothing(
                    x_sample_padded).squeeze(dim=1)
                smooth_loss_train = mse_dis(x_sample, spec_smoothed)
                smooth_loss_train.backward()
                smooth_optimizer.step()

                # Init gradients
                self.zerograd()


            ### Validation ###
            avg_mutual_info /= n_batch
            
            self.encoder.eval()
            self.decoder.eval()
            self.discriminator.eval()
            spec_in, aux_in = [torch.cat(x, dim=0)
                               for x in zip(*list(self.val_loader))]
            if self.train_loader.dataset.aux is None:
                aux_in = None
            else:
                assert len(aux_in.size()) == 2
                n_aux = aux_in.size()[-1]
                aux_in = aux_in.to(self.device)
            spec_in = spec_in.to(self.device)
            z = self.encoder(spec_in)
            spec_re = self.decoder(z)
            recon_loss_val = mse_dis(spec_re, spec_in)

                
            if self.train_loader.dataset.aux is not None:
                aux_target = torch.sign(aux_in[:, np.newaxis, :] - aux_in[np.newaxis, :, :])
                z_aux = z[:, :n_aux]
                assert len(z_aux.size()) == 2
                aux_pred = z_aux[:, np.newaxis, :] - z_aux[np.newaxis, :, :]
                aux_len = aux_pred.size()[0]
                aux_loss_val = - (aux_pred * aux_target).sum() / ((aux_len**2 - aux_len) * n_aux)
            else:
                aux_loss_val = None

            style_np = z.detach().clone().cpu().numpy().T
            style_shapiro = [shapiro(x).statistic for x in style_np]

            style_coupling = np.max(np.fabs([spearmanr(style_np[j1], style_np[j2]).correlation
                                             for j1, j2 in itertools.combinations(range(style_np.shape[0]), 2)]))

            z_fake_gauss = z
            z_real_gauss = torch.randn_like(
                z, requires_grad=True, device=self.device)
            real_gauss_label = torch.ones(
                spec_in.size()[0], dtype=torch.long, requires_grad=False, device=self.device)
            real_gauss_pred = self.discriminator(z_real_gauss, alpha)
            fake_guass_lable = torch.zeros(
                spec_in.size()[0], dtype=torch.long, requires_grad=False, device=self.device)
            fake_gauss_pred = self.discriminator(z_fake_gauss, alpha)

            adversarial_loss_val = nll_loss(
                real_gauss_pred, real_gauss_label) + nll_loss(fake_gauss_pred, fake_guass_lable)

            x_sample = self.decoder(z)
            x_sample_padded = self.padding4smooth(
                x_sample.unsqueeze(dim=1))
            spec_smoothed = self.gaussian_smoothing(
                x_sample_padded).squeeze(dim=1)
            smooth_loss_val = mse_dis(x_sample, spec_smoothed)

            z = torch.randn(self.batch_size, self.nstyle,
                            requires_grad=False, device=self.device)
            x_sample = self.decoder(z)
            z_recon = self.encoder(x_sample)
            mutual_info_loss_val = mse_dis(z_recon, z)

            # write losses to tensorboard
            if self.verbose:
                self.tb_writer.add_scalars("Adversarial", 
                    {
                        'Train': adversarial_loss_train.item(),
                        'Validation': adversarial_loss_val.item()
                    },
                    global_step = epoch
                )
                self.tb_writer.add_scalars("Aux", 
                    {
                        'Train': aux_loss_train.item(),
                        'Validation': aux_loss_val.item()
                    },
                    global_step = epoch
                )
                self.tb_writer.add_scalars("Recon", 
                    {
                        'Train': recon_loss_train.item(),
                        'Validation': recon_loss_val.item()
                    },
                    global_step = epoch
                )
                self.tb_writer.add_scalars("Smooth", 
                    {
                        'Train': smooth_loss_train.item(),
                        'Validation': smooth_loss_val.item()
                    },
                    global_step = epoch
                )
                self.tb_writer.add_scalars("Mutual Info", 
                    {
                        'Train': mutual_info_loss_train.item(),
                        'Validation': mutual_info_loss_val.item()
                    },
                    global_step = epoch
                )

            model_dict = {"Encoder": self.encoder,
                          "Decoder": self.decoder,
                          "Style Discriminator": self.discriminator}
            metrics = [min(style_shapiro), recon_loss_val.item(), avg_mutual_info, style_coupling,
                       aux_loss_val.item() if aux_in is not None else 0]
            
            combined_metric = - (np.array(self.metric_weights) * np.array(metrics)).sum()
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_chpt_file = f"{chkpt_dir}/epoch_{epoch:06d}_loss_{combined_metric:07.6g}.pt"
                torch.save(model_dict, best_chpt_file)

            for sch in schedulers:
                sch.step(combined_metric)

            if callback is not None:
                callback(epoch, metrics)
            
            # plot images
            if epoch % 25 == 0:
                spec_in = [torch.cat(x, dim=0)
                           for x in zip(*list(self.val_loader))][0]
                spec_in = spec_in.to(self.device)
                z = self.encoder(spec_in)
                if self.verbose:
                    self.tb_writer.add_figure("Style Value Distribution", 
                        self.get_style_distribution_plot(z.clone().cpu().detach().numpy()),
                        global_step = epoch
                    )

        # save the final model
        torch.save(model_dict, f'{self.work_dir}/final.pt')

        if best_chpt_file is not None:
            shutil.copy2(best_chpt_file, f'{self.work_dir}/best.pt')

        return metrics

    @classmethod
    def from_data(
        cls, csv_fn, 
        igpu=0, max_epoch=2000, verbose=True, work_dir='.', 
        train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, 
        config_parameters = Parameters({}),
        logger = logging.getLogger("from_data")
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
            n_layers = p.nlayers
        )
        decoder = AE_CLS_DICT[p.ae_form]["decoder"](
            nstyle = p.nstyle, 
            dropout_rate = p.dropout_rate, 
            last_layer_activation = p.decoder_activation, 
            dim_out = p.dim_out,
            n_layers = p.nlayers
        )
        if p.use_cnn_discriminator:
            discriminator = DiscriminatorCNN(
                nstyle=p.nstyle, dropout_rate=p.grad_rev_dropout_rate, noise=p.grad_rev_noise
            )
        else:
            discriminator = DiscriminatorFC(
                nstyle=p.nstyle, dropout_rate=p.grad_rev_dropout_rate, noise=p.grad_rev_noise,
                layers = p.FC_discriminator_layers
            )

        for net in [encoder, decoder, discriminator]:
            net.to(device)

        # Load trainer
        trainer = Trainer(
            encoder, decoder, discriminator, device, dl_train, dl_val,
            max_epoch=max_epoch, verbose=verbose, work_dir=work_dir,
            config_parameters = p, logger = logger
        )
        return trainer


    @staticmethod
    def test_models(csv_fn, n_aux=0,
                    train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15, work_dir='.',
                    final_model_name='final.pt', best_model_name='best.pt'):
        final_spuncat = torch.load(
            f'{work_dir}/{final_model_name}', map_location=torch.device('cpu'))

        transform_list = transforms.Compose([ToTensor()])
        _, _, dataset_test = [AuxSpectraDataset(
            csv_fn, p, (train_ratio, validation_ratio, test_ratio),
            transform=transform_list, n_aux=n_aux)
            for p in ["train", "val", "test"]]

        def plot_style_distributions(encoder, ds, title_base="Style Distribution"):
            encoder.eval()
            spec_in = torch.tensor(ds.spec.copy(), dtype=torch.float32)
            z = encoder(spec_in).clone().detach().cpu().numpy()
            nstyle = z.shape[1]
            # noinspection PyTypeChecker
            fig, ax_list = plt.subplots(
                nstyle, 1, sharex=True, sharey=True, figsize=(9, 12))
            for istyle, ax in zip(range(nstyle), ax_list):
                sns.histplot(z[:, istyle], kde=False, color='blue', bins=np.arange(-3.0, 3.01, 0.2),
                             ax=ax, element="step")
                ax.set_xlabel(f"Style #{istyle}")
                ax.set_ylabel("Counts")

            title = f'{title_base}'
            fig.suptitle(title, y=0.91)

            report_dir = os.path.join(work_dir, "reports")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            plt.savefig(f'{report_dir}/{title}.pdf',
                        dpi=300, bbox_inches='tight')

        plot_style_distributions(final_spuncat["Encoder"], dataset_test,
                                 title_base="Style Distribution on FEFF Test Set")
