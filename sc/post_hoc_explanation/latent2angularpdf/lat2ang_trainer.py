import itertools
from numpy.lib.arraysetops import isin
from sklearn import metrics

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch_optimizer as ex_optim
import shutil
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sc.post_hoc_explanation.latent2angularpdf.lat2ang_dataloader import get_latent2apdf_dataloaders
from sc.post_hoc_explanation.latent2angularpdf.lat2ang_model import Latent2AngularPDF


class Latent2AngularPDFTrainer:

    def __init__(self, model, device, train_loader, val_loader,
                 lr=1.0E-3, max_epoch=300,  sch_factor=0.25, sch_patience=300, style_noise=0.01, weight_decay=1e-2,
                 nclasses=3, nstyle=2, ntest_image_per_style = 11, image_dim=(64, 64), plot_interval=25,
                 optimizer_name="AdamW", verbose=True, tb_logdir="runs", work_dir='.'):
        self.device = device
        self.model: Latent2AngularPDF = model.to(self.device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.lr = lr
        self.max_epoch = max_epoch
        self.sch_factor = sch_factor
        self.sch_patience = sch_patience
        self.style_noise = style_noise
        self.weight_deca = weight_decay
        self.optimizer_name = optimizer_name
        self.nclasses = nclasses
        self.nstyle = nstyle
        self.ntest_image_per_style = ntest_image_per_style
        self.image_dim = tuple(image_dim)
        self.plot_interval = plot_interval
        self.verbose = verbose

        if verbose:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(work_dir, tb_logdir))
            example_lat = iter(self.train_loader).next()[0]
            with torch.no_grad():
                self.tb_writer.add_graph(self.model, example_lat)
    
    def precompute_latent_variation(self):
        one_hot_variation = torch.eye(self.nclasses).unsqueeze(dim=1).repeat([1, self.nstyle, 1])\
            .unsqueeze(dim=2).repeat([1, 1, self.ntest_image_per_style, 1])
        style_variation = torch.zeros([self.nclasses, self.nstyle, self.ntest_image_per_style, self.nstyle])
        style_variation[:, torch.arange(self.nstyle), :, torch.arange(self.nstyle)] = torch.linspace(-2, 2, self.ntest_image_per_style)
        latent_variation = torch.cat([one_hot_variation, style_variation], dim=-1)
        latent_variation = latent_variation.reshape([self.nclasses * self.nstyle * self.ntest_image_per_style, 
                                                     self.nclasses + self.nstyle])
        return latent_variation

    def get_style_image_variation_plot(self, image_list: np.ndarray):
        assert isinstance(image_list, np.ndarray)
        image_list = image_list.reshape((self.nclasses, self.nstyle, self.ntest_image_per_style) + self.image_dim)
        fig_dict = dict()
        for iclass in range(len(self.nclasses)):
            # noinspection PyTypeChecker
            fig, ax_list = plt.subplots(self.nstyle, 1, sharex=True, sharey=True, figsize=(4, 4))
            for jstyle, (image_row, ax_row) in enumerate(zip(image_list[iclass], ax_list)):
                for ktest, (image, ax) in enumerate(zip(image_row, ax_row)):
                    plt.imshow(image, extent=[1.6, 8.0, 180, 0], aspect=0.035, interpolation='spline36', cmap='jet')
                    if ktest == 0:
                        ax.ylabel("$\\theta$")
                    if jstyle == self.nstyle - 1:
                        ax.xlabel('r ($\AA$)')
            title = f"Image Variation CN{iclass + 4}"
            fig.suptitle(title, y=0.91)
            fig_dict[title] = fig
        return fig_dict
 
    def train(self, callback=None):
        if self.verbose:
            para_info = torch.__config__.parallel_info()
            print(para_info)

        opt_cls_dict = {"Adam": optim.Adam, "AdamW": optim.AdamW,
                        "AdaBound": ex_optim.AdaBound, "RAdam": ex_optim.RAdam}
        opt_cls = opt_cls_dict[self.optimizer_name]

        # loss function
        mse_dis_mean = nn.MSELoss().to(self.device)
        mse_dis_sum = nn.MSELoss(reduction='sum').to(self.device)
        reconn_solver: optim.Optimizer = opt_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(reconn_solver, factor=self.sch_factor, patience=self.sch_patience, cooldown=0, threshold=0.01, verbose=self.verbose)
        latent_variation = self.precompute_latent_variation().to(self.device)

        last_best = 0.0
        chkpt_dir = f"{self.work_dir}/checkpoints"
        best_chk = None
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir, exist_ok=True)

        for epoch in range(self.max_epoch):
            self.model.train()
            for lat, apdf_in in self.train_loader:
                lat = lat.to(self.device)
                apdf_in = apdf_in.to(self.device)

                reconn_solver.zero_grad()
                apdf_pred = self.model(apdf_in)
                reconn_loss: torch.Tensor = mse_dis_mean(apdf_in, apdf_pred)
                reconn_loss.backward()
                reconn_solver.step()

                if self.verbose:
                    # record losses
                    loss_dict = {
                        'Reconn': reconn_loss.item(),
                    }
                    self.tb_writer.add_scalars("Loss/train", loss_dict, global_step=epoch)
            
            self.model.eval()
            val_loss_list = []
            for lat, apdf_in in self.val_loader:
                lat = lat.to(self.device)
                apdf_in = apdf_in.to(self.device)
                apdf_pred = self.model(apdf_in)
                reconn_loss: torch.Tensor = mse_dis_sum(apdf_in, apdf_pred)
                val_loss_list.append(reconn_loss.item())
            val_mean_loss = sum(val_loss_list)/len(self.val_loader.dataset)
            metrics = [val_mean_loss]
            if self.verbose:
                # record losses
                loss_dict = {
                    'Reconn': val_mean_loss,
                }
                self.tb_writer.add_scalars("Loss/val", loss_dict, global_step=epoch)

            scheduler.step(val_mean_loss)

            if val_mean_loss < last_best * 0.99:
                chk_fn = f"{chkpt_dir}/epoch_{epoch:06d}_loss_{val_mean_loss:05.4g}.pt"
                torch.save(self.model, chk_fn)
                last_best = val_mean_loss
                best_chk = chk_fn
            
            if epoch % self.plot_interval == 0 and self.verbose:
                image_list = self.model(latent_variation)
                image_list = image_list.clone().detach().cpu().numpy()
                fig_dict = self.get_style_image_variation_plot(image_list)
                for title, fig in fig_dict.items():
                        self.tb_writer.add_figure(title, fig, global_step=epoch)

        torch.save(self.model, f'{self.work_dir}/final.pt')
        if best_chk is not None:
            shutil.copy2(best_chk, f'{self.work_dir}/best.pt')

        return metrics
