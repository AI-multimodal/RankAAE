import itertools

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import torch_optimizer as ex_optim
import shutil
import os
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sc.clustering.model import CompactDecoder, CompactEncoder, Encoder, Decoder, GaussianSmoothing, DummyDualAAE, DiscriminatorCNN, DiscriminatorFC
from sc.clustering.dataloader import get_dataloaders
from torchvision import transforms
from sc.clustering.dataloader import AuxSpectraDataset, ToTensor
from scipy.stats import shapiro, spearmanr


class Trainer:

    def __init__(self, encoder, decoder, discriminator, device, train_loader, val_loader,
                 base_lr=0.0001, nstyle=2,
                 batch_size=111, max_epoch=300,
                 tb_logdir="runs", use_cnn_dis=False,
                 grad_rev_beta=1.1, alpha_flat_step=100, alpha_limit=2.0,
                 sch_factor=0.25, sch_patience=300, spec_noise=0.01, weight_decay=1e-2,
                 lr_ratio_Reconn=2.0, lr_ratio_Mutual=3.0, lr_ratio_Smooth=0.1,
                 lr_ratio_Corr=0.5, lr_ratio_Style=0.5, optimizer_name="AdamW",
                 verbose=True, work_dir='.', use_flex_spec_target=False):

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.discriminator = discriminator.to(device)
        self.nstyle = nstyle
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.base_lr = base_lr
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.noise_test_range = (-2, 2)
        self.ntest_per_spectra = 10
        gau_kernel_size = 17
        self.gaussian_smoothing = GaussianSmoothing(channels=1, kernel_size=gau_kernel_size, sigma=3.0, dim=1,
                                                    device=device).to(device)
        self.padding4smooth = nn.ReplicationPad1d(
            padding=(gau_kernel_size - 1) // 2).to(device)

        if verbose:
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(work_dir, tb_logdir))
            example_spec = iter(train_loader).next()[0]
            self.tb_writer.add_graph(DummyDualAAE(
                use_cnn_dis, self.encoder.__class__, self.decoder.__class__), example_spec)

        self.grad_rev_beta = grad_rev_beta
        self.alpha_flat_step = alpha_flat_step
        self.alpha_limit = alpha_limit
        self.sch_factor = sch_factor
        self.sch_patience = sch_patience
        self.spec_noise = spec_noise
        self.lr_ratio_Reconn = lr_ratio_Reconn
        self.lr_ratio_Mutual = lr_ratio_Mutual
        self.lr_ratio_Smooth = lr_ratio_Smooth
        self.lr_ratio_Corr = lr_ratio_Corr
        self.lr_ratio_Style = lr_ratio_Style
        self.verbose = verbose
        self.work_dir = work_dir
        self.use_flex_spec_target = use_flex_spec_target
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

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
            logging.info(para_info)

        opt_cls_dict = {"Adam": optim.Adam, "AdamW": optim.AdamW,
                        "AdaBound": ex_optim.AdaBound, "RAdam": ex_optim.RAdam}
        opt_cls = opt_cls_dict[self.optimizer_name]

        # loss function
        mse_dis = nn.MSELoss().to(self.device)
        nll_loss = nn.NLLLoss().to(self.device)

        reconn_solver = opt_cls([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
                                lr=self.lr_ratio_Reconn * self.base_lr,
                                weight_decay=self.weight_decay)
        mutual_info_solver = opt_cls([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
                                     lr=self.lr_ratio_Mutual * self.base_lr)
        smooth_solver = opt_cls([{'params': self.decoder.parameters()}], lr=self.lr_ratio_Smooth * self.base_lr,
                                weight_decay=self.weight_decay)
        corr_solver = opt_cls([{'params': self.encoder.parameters()}], lr=self.lr_ratio_Corr * self.base_lr,
                              weight_decay=self.weight_decay)
        adversarial_solver = opt_cls([{'params': self.discriminator.parameters()},
                                      {'params': self.encoder.parameters()}],
                                     lr=self.lr_ratio_Style * self.base_lr,
                                     betas=(self.grad_rev_beta * 0.9, self.grad_rev_beta * 0.009 + 0.99))

        sol_list = [reconn_solver, mutual_info_solver, smooth_solver,
                     corr_solver, adversarial_solver]
        schedulers = [
            ReduceLROnPlateau(sol, mode="min", factor=self.sch_factor, patience=self.sch_patience, cooldown=0, threshold=0.01,
                              verbose=self.verbose)
            for sol in sol_list]

        # train network
        last_best = 10.0
        chkpt_dir = f"{self.work_dir}/checkpoints"
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir, exist_ok=True)
        best_chk = None
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
                adversarial_loss = nll_loss(real_gauss_pred, real_gauss_label) + \
                    nll_loss(fake_gauss_pred, fake_guass_lable)
                adversarial_loss.backward()
                adversarial_solver.step()

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
                    aux_loss = - (aux_pred * aux_target).sum() / ((aux_len**2 - aux_len) * n_aux)
                    aux_loss.backward()
                    corr_solver.step()
                else:
                    aux_loss = None

                # Init gradients, reconstruction loss
                self.zerograd()
                z = self.encoder(spec_in)
                spec_re = self.decoder(z)
                if not self.use_flex_spec_target:
                    recon_loss = mse_dis(spec_re, spec_target)
                else:
                    spec_scale = torch.abs(spec_re.mean(
                        dim=1)) / torch.abs(spec_target.mean(dim=1))
                    recon_loss = ((spec_scale - 1.0) ** 2).mean() * 0.1
                    spec_scale = spec_scale.detach()
                    spec_scale = torch.clamp(spec_scale, min=0.7, max=1.3)
                    recon_loss += mse_dis(spec_re,
                                          (spec_target.T * spec_scale).T)
                recon_loss.backward()
                reconn_solver.step()

                # Init gradients, mutual information loss
                self.zerograd()
                z = torch.randn(self.batch_size, self.nstyle,
                                requires_grad=False, device=self.device)
                x_sample = self.decoder(z)
                z_recon = self.encoder(x_sample)
                mutual_info_loss = mse_dis(z_recon, z)
                mutual_info_loss.backward()
                mutual_info_solver.step()
                avg_mutual_info += mutual_info_loss.item()

                # Init gradients, smoothness loss
                self.zerograd()
                x_sample = self.decoder(z)
                x_sample_padded = self.padding4smooth(
                    x_sample.unsqueeze(dim=1))
                spec_smoothed = self.gaussian_smoothing(
                    x_sample_padded).squeeze(dim=1)
                Smooth_loss = mse_dis(x_sample, spec_smoothed)
                Smooth_loss.backward()
                smooth_solver.step()

                # Init gradients
                self.zerograd()

                if self.verbose:
                    # record losses
                    loss_dict = {
                        'Recon': recon_loss.item(),
                        'Mutual Info': mutual_info_loss.item(),
                        'Smooth': Smooth_loss.item()
                    }

                    self.tb_writer.add_scalars(
                        "Recon/train", loss_dict, global_step=epoch)
                    if self.train_loader.dataset.aux is not None:
                        loss_dict = {"Aux": aux_loss.item()}
                        self.tb_writer.add_scalars(
                            "Aux/train", loss_dict, global_step=epoch)
                    loss_dict = {
                        'Adversarial': adversarial_loss.item()
                    }
                    self.tb_writer.add_scalars(
                        "Adversarial/train", loss_dict, global_step=epoch)

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
            recon_loss = mse_dis(spec_re, spec_in)
            loss_dict = {
                'Recon': recon_loss.item()
            }
            if self.verbose:
                self.tb_writer.add_scalars(
                    "Recon/val", loss_dict, global_step=epoch)
            if self.train_loader.dataset.aux is not None:
                aux_target = torch.sign(aux_in[:, np.newaxis, :] - aux_in[np.newaxis, :, :])
                z_aux = z[:, :n_aux]
                assert len(z_aux.size()) == 2
                aux_pred = z_aux[:, np.newaxis, :] - z_aux[np.newaxis, :, :]
                aux_len = aux_pred.size()[0]
                aux_loss = - (aux_pred * aux_target).sum() / ((aux_len**2 - aux_len) * n_aux)
                loss_dict = {"Aux": aux_loss.item()}
                if self.verbose:
                    self.tb_writer.add_scalars("Aux/val", loss_dict, global_step=epoch)
            else:
                aux_loss = None

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

            adversarial_loss = nll_loss(
                real_gauss_pred, real_gauss_label) + nll_loss(fake_gauss_pred, fake_guass_lable)
            loss_dict = {
                'Adversarial': adversarial_loss.item()
            }
            if self.verbose:
                self.tb_writer.add_scalars(
                    "Adversarial/val", loss_dict, global_step=epoch)

            model_dict = {"Encoder": self.encoder,
                          "Decoder": self.decoder,
                          "Style Discriminator": self.discriminator}

            metrics = [min(style_shapiro), recon_loss.item(), avg_mutual_info, style_coupling,
                       aux_loss.item() if aux_in is not None else 0]
            
            if callback is not None:
                combined_metric = - (np.array(callback.metric_weights) * np.array(metrics)).sum()
            else:
                combined_metric = metrics[3] # use style_coupling only
            if abs(combined_metric) > abs(last_best) * 1.01:
                chk_fn = f"{chkpt_dir}/epoch_{epoch:06d}_loss_{combined_metric:07.6g}.pt"
                torch.save(model_dict, chk_fn)
                best_chk = chk_fn
                last_best = combined_metric

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
                    fig = self.get_style_distribution_plot(
                        z.clone().cpu().detach().numpy())
                    self.tb_writer.add_figure(
                        "Style Value Distribution", fig, global_step=epoch)

        # save model
        model_dict = {"Encoder": self.encoder,
                      "Decoder": self.decoder,
                      "Style Discriminator": self.discriminator}
        torch.save(model_dict,
                   f'{self.work_dir}/final.pt')
        if best_chk is not None:
            shutil.copy2(best_chk, f'{self.work_dir}/best.pt')

        return metrics

    @classmethod
    def from_data(cls, csv_fn, igpu=0, batch_size=512, lr=0.01, max_epoch=2000, nstyle=2,
                  dropout_rate=0.5, grad_rev_dropout_rate=0.5, grad_rev_noise=0.1, grad_rev_beta=1.1,
                  use_cnn_dis=False, alpha_flat_step=100, alpha_limit=2.0,
                  sch_factor=0.25, sch_patience=300, spec_noise=0.01,
                  lr_ratio_Reconn=2.0, lr_ratio_Mutual=3.0, lr_ratio_Smooth=0.1,
                  lr_ratio_Style=0.5, lr_ratio_Corr=0.5, weight_decay=1e-2,
                  train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15,
                  use_flex_spec_target=False, optimizer_name="AdamW",
                  decoder_activation='Softplus', ae_form='compact', n_aux=0,
                  verbose=True, work_dir='.'):
        ae_cls_dict = {"normal": {"encoder": Encoder, "decoder": Decoder},
                       "compact": {"encoder": CompactEncoder, "decoder": CompactDecoder}}
        assert ae_form in ae_cls_dict
        dl_train, dl_val, dl_test = get_dataloaders(
            csv_fn, batch_size, (train_ratio, validation_ratio, test_ratio), n_aux=n_aux)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            if verbose:
                logging.info("Use GPU")
            for loader in [dl_train, dl_val]:
                loader.pin_memory = False
        else:
            if verbose:
                logging.warn("Use Slow CPU!")

        device = torch.device(f"cuda:{igpu}" if use_cuda else "cpu")

        encoder = ae_cls_dict[ae_form]["encoder"](
            nstyle=nstyle, dropout_rate=dropout_rate)
        decoder = ae_cls_dict[ae_form]["decoder"](
            nstyle=nstyle, dropout_rate=dropout_rate, last_layer_activation=decoder_activation)
        if use_cnn_dis:
            discriminator = DiscriminatorCNN(
                nstyle=nstyle, dropout_rate=grad_rev_dropout_rate, noise=grad_rev_noise)
        else:
            discriminator = DiscriminatorFC(
                nstyle=nstyle, dropout_rate=grad_rev_dropout_rate, noise=grad_rev_noise)

        for i in [encoder, decoder, discriminator]:
            i.to(device)

        trainer = Trainer(encoder, decoder, discriminator, device, dl_train, dl_val,
                          nstyle=nstyle, weight_decay=weight_decay,
                          max_epoch=max_epoch, base_lr=lr, use_cnn_dis=use_cnn_dis,
                          grad_rev_beta=grad_rev_beta, alpha_flat_step=alpha_flat_step, alpha_limit=alpha_limit,
                          sch_factor=sch_factor, sch_patience=sch_patience, spec_noise=spec_noise,
                          lr_ratio_Reconn=lr_ratio_Reconn, lr_ratio_Mutual=lr_ratio_Mutual,
                          lr_ratio_Smooth=lr_ratio_Smooth, lr_ratio_Corr=lr_ratio_Corr,
                          lr_ratio_Style=lr_ratio_Style, optimizer_name=optimizer_name,
                          use_flex_spec_target=use_flex_spec_target,
                          verbose=verbose, work_dir=work_dir)
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
