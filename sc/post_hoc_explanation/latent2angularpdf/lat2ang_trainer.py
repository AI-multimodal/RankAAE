import itertools
from sc.clustering.trainer import Trainer
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
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.nclasses = nclasses
        self.nstyle = nstyle
        self.ntest_image_per_style = ntest_image_per_style
        self.image_dim = tuple(image_dim)
        self.plot_interval = plot_interval
        self.verbose = verbose
        self.work_dir = work_dir

        if verbose:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(work_dir, tb_logdir))
            example_lat = iter(self.train_loader).next()[0]
            example_lat = example_lat.cpu()
            with torch.no_grad():
                self.tb_writer.add_graph(self.model.cpu(), example_lat)
            self.model.to(self.device)
    
    @staticmethod
    def precompute_latent_variation(nclasses, nstyle, ntest_image_per_style):
        one_hot_variation = torch.eye(nclasses).unsqueeze(dim=1).repeat([1, nstyle, 1])\
            .unsqueeze(dim=2).repeat([1, 1, ntest_image_per_style, 1])
        style_variation = torch.zeros([nclasses, nstyle, ntest_image_per_style, nstyle])
        style_variation[:, torch.arange(nstyle), :, torch.arange(nstyle)] = torch.linspace(-2, 2, ntest_image_per_style)
        latent_variation = torch.cat([one_hot_variation, style_variation], dim=-1)
        latent_variation = latent_variation.reshape([nclasses * nstyle * ntest_image_per_style, 
                                                     nclasses + nstyle])
        return latent_variation

    @staticmethod
    def get_style_image_variation_plot(image_list: np.ndarray, nclasses, nstyle, ntest_image_per_style, image_dim=(64, 64), base_title=""):
        assert isinstance(image_list, np.ndarray)
        image_list = image_list.reshape((nclasses, nstyle, ntest_image_per_style) + image_dim)
        fig_dict = dict()
        for iclass in range(nclasses):
            # noinspection PyTypeChecker
            fig, ax_list = plt.subplots(nstyle, ntest_image_per_style, sharex=True, sharey=True, figsize=(20, 4))
            for jstyle, (image_row, ax_row) in enumerate(zip(image_list[iclass], ax_list)):
                for ktest, (image, ax) in enumerate(zip(image_row, ax_row)):
                    ax.imshow(image, extent=[1.6, 8.0, 180, 0], aspect=0.035, interpolation='spline36', cmap='jet')
                    if ktest == 0:
                        ax.set_ylabel(f"Style {jstyle + 1}\n" + "$\\theta$")
                    if jstyle == nstyle - 1:
                        ax.set_xlabel('r ($\AA$)')
            title = f"{base_title}Image Variation CN{iclass + 4}"
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
        mse_dis_sum = nn.MSELoss(reduction='none').to(self.device)
        reconn_solver: optim.Optimizer = opt_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(reconn_solver, factor=self.sch_factor, patience=self.sch_patience, cooldown=0, threshold=0.01, 
                                      mode="min", verbose=self.verbose)
        latent_variation = self.precompute_latent_variation(self.nclasses, self.nstyle, self.ntest_image_per_style)
        latent_variation = latent_variation.to(self.device)

        last_best = 99999999999.0
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
                lat_noise = torch.randn([lat.size()[0], self.nstyle], device=self.device) * self.style_noise
                lat_pert = lat.clone()
                lat_pert[:, -self.nstyle:] += lat_noise
                apdf_pred = self.model(lat_pert)
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
                apdf_pred = self.model(lat)
                reconn_loss: torch.Tensor = mse_dis_sum(apdf_in, apdf_pred)
                val_loss_list.append(reconn_loss)
            val_mean_loss = torch.cat(val_loss_list, dim=0).mean().item()
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
            
            if epoch % self.plot_interval == 0 and epoch > 1 and self.verbose:
                image_list = self.model(latent_variation)
                image_list = image_list.clone().detach().cpu().numpy()
                fig_dict = self.get_style_image_variation_plot(image_list, self.nclasses, self.nstyle, self.ntest_image_per_style, self.image_dim)
                for title, fig in fig_dict.items():
                        self.tb_writer.add_figure(title, fig, global_step=epoch)

        torch.save(self.model, f'{self.work_dir}/final.pt')
        if best_chk is not None:
            shutil.copy2(best_chk, f'{self.work_dir}/best.pt')

        return metrics

    @classmethod
    def from_data(cls, pkl_fn, igpu=0, batch_size=512, lr=1.0E-3, max_epoch=300,  sch_factor=0.25, sch_patience=300, 
                  style_noise=0.01, weight_decay=1e-2, optimizer_name="AdamW", verbose=True, work_dir='.',
                  dropout_rate=0.05):
        dl_train, dl_val, _ = get_latent2apdf_dataloaders(pkl_fn, batch_size)
        
        use_cuda = torch.cuda.is_available() and igpu >= 0
        if use_cuda:
            if verbose:
                print("Use GPU")
            for loader in [dl_train, dl_val]:
                loader.pin_memory = False
        else:
            if verbose:
                print("Use Slow CPU!")

        device = torch.device(f"cuda:{igpu}" if use_cuda else "cpu")
        model = Latent2AngularPDF(lat_size=5, dropout_rate=dropout_rate)
        model = model.to(device)
        trainer = Latent2AngularPDFTrainer(model, device, dl_train, dl_val, lr=lr, max_epoch=max_epoch, sch_factor=sch_factor,
                                           sch_patience=sch_patience, style_noise=style_noise, weight_decay=weight_decay, 
                                           optimizer_name=optimizer_name, verbose=verbose, work_dir=work_dir)
        return trainer

    @classmethod
    def test_models(cls, work_dir='.', final_model_name='final.pt', best_model_name='best.pt',
                    nclasses=3, nstyle=2, ntest_image_per_style = 11, image_dim=(64, 64)):
        final_model = torch.load(f'{work_dir}/{final_model_name}', map_location=torch.device('cpu'))
        best_model = torch.load(f'{work_dir}/{best_model_name}', map_location=torch.device('cpu'))

        def plot_variation(model, title='final '):
            model.eval()
            latent = cls.precompute_latent_variation(nclasses, nstyle, ntest_image_per_style)
            image_list = model(latent).cpu().detach().numpy()
            fig_dict = cls.get_style_image_variation_plot(image_list, nclasses, nstyle, ntest_image_per_style, image_dim=image_dim, base_title=title)
            report_dir = os.path.join(work_dir, "reports")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
            for title, fig in fig_dict.items():
                fig.savefig(f'{report_dir}/{title}.pdf', dpi=300)

        plot_variation(final_model, title='Final ')
        plot_variation(best_model, title='Best ')