import os
import dlib
import torch
import torch.nn as nn

import numpy as np
import torchvision

from functools import partial
from sklearn.decomposition import IncrementalPCA
from core.utils.image_utils import BicubicDownSample
from core.dataset import ImagesDataset
from core.loss import LossBuilder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from gan_models.StyleGAN2.model import Generator

toPIL = torchvision.transforms.ToPILImage()


class IPCAEstimator:
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(
            n_components, whiten=self.whiten, batch_size=max(100, 5 * n_components)
        )
        self.batch_support = True

    def get_param_str(self):
        return "ipca_c{}{}".format(self.n_components, "_w" if self.whiten else "")

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            self.transformer.n_samples_seen_ = self.transformer.n_samples_seen_.astype(
                np.int64
            )  # avoid overflow
            return True
        except ValueError as e:
            print(f"\nIPCA error:", e)
            return False

    def get_components(self):
        stdev = np.sqrt(self.transformer.explained_variance_)  # already sorted
        var_ratio = self.transformer.explained_variance_ratio_
        return (
            self.transformer.components_,
            stdev,
            var_ratio,
        )  # PCA outputs are normalized


class Net(nn.Module):
    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.generator = Generator(
            opts.size,
            opts.latent,
            opts.n_mlp,
            channel_multiplier=opts.channel_multiplier,
        )
        self.cal_layer_num()
        self.load_weights()
        self.load_PCA_model()

    def load_weights(self):
        print("Loading StyleGAN2 from checkpoint: {}".format(self.opts.ckpt))
        checkpoint = torch.load(self.opts.ckpt)
        device = self.opts.device
        self.generator.load_state_dict(checkpoint["g_ema"])
        self.latent_avg = checkpoint["latent_avg"]
        self.generator.to(device)
        self.latent_avg = self.latent_avg.to(device)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

    def build_PCA_model(self, PCA_path):

        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32)
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu()
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
            self.generator.style.to(self.opts.device)

        transformer = IPCAEstimator(512)
        X_mean = pulse_space.mean(0)
        transformer.fit(pulse_space - X_mean)
        X_comp, X_stdev, X_var_ratio = transformer.get_components()
        np.savez(
            PCA_path,
            X_mean=X_mean,
            X_comp=X_comp,
            X_stdev=X_stdev,
            X_var_ratio=X_var_ratio,
        )

    def load_PCA_model(self):
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + "_PCA.npz"

        if not os.path.isfile(PCA_path):
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model["X_mean"]).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model["X_comp"]).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model["X_stdev"]).float().to(device)

    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_layer_num(self):
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14
        return

    def cal_p_norm_loss(self, latent_in):
        latent_p_norm = (
            torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean
        ).bmm(self.X_comp.T.unsqueeze(0)) / self.X_stdev
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())
        return p_norm_loss


class II2S(nn.Module):
    def __init__(self, opts):
        super(II2S, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.setup_loss_builder()

        # self.image_transform = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
    
    def setup_loss_builder(self):
        self.loss_builder = LossBuilder(self.opts)
    
    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)

    def setup_optimizer(self):
        opt_dict = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "sgdm": partial(torch.optim.SGD, momentum=0.9),
            "adamax": torch.optim.Adamax,
        }

        latent = []
        if self.opts.tile_latent:
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer, latent

    def invert_images(
        self,
        image_path=None,
        output_dir=None,
        return_latents=False,
        align_input=False,
        save_output=True,
    ):

        final_latents = None
        if return_latents:
            final_latents = []

        self.setup_dataloader(image_path=image_path, align_input=align_input)
        device = self.opts.device

        # ibar = tqdm(self.dataloader, desc='Images')
        # for ref_im_H, ref_im_L, ref_name in ibar:

        for ref_im_H, ref_im_L, ref_name in self.dataloader:
            optimizer, latent = self.setup_optimizer()
            pbar = tqdm(range(self.opts.steps), desc="Embedding")
            for step in pbar:
                optimizer.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)

                gen_im, _ = self.net.generator(
                    [latent_in], input_is_latent=True, return_latents=False
                )
                im_dict = {
                    "ref_im_H": ref_im_H.to(device),
                    "ref_im_L": ref_im_L.to(device),
                    "gen_im_H": gen_im,
                    "gen_im_L": self.downsample(gen_im),
                }
                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer.step()

                if self.opts.verbose:
                    pbar.set_description(
                        "Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}".format(
                            loss, loss_dic["l2"], loss_dic["percep"], loss_dic["p-norm"]
                        )
                    )

                if (
                    self.opts.save_intermediate
                    and step % self.opts.save_interval == 0
                    and save_output
                ):
                    self.save_intermediate_results(
                        ref_name, gen_im, latent_in, step, output_dir
                    )

            if save_output:
                self.save_results(ref_name, gen_im, latent_in, output_dir)

            if return_latents:
                final_latents.append(latent_in)

        return final_latents

    def invert_image(
        self,
        ref_im_H,
        ref_im_L
    ):

        final_latents = []
        optimizer, latent = self.setup_optimizer()
        pbar = tqdm(range(self.opts.steps), desc="II2S invert processing")

        for step in pbar:
            optimizer.zero_grad()
            latent_in = torch.stack(latent).unsqueeze(0)

            gen_im, _ = self.net.generator(
                [latent_in], input_is_latent=True, return_latents=False
            )
            im_dict = {
                "ref_im_H": ref_im_H.to(self.opts.device),
                "ref_im_L": ref_im_L.to(self.opts.device),
                "gen_im_H": gen_im,
                "gen_im_L": self.downsample(gen_im),
            }
            loss, loss_dic = self.cal_loss(im_dict, latent_in)
            loss.backward()
            optimizer.step()

            if self.opts.verbose:
                pbar.set_description(
                    "Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}".format(
                        loss, loss_dic["l2"], loss_dic["percep"], loss_dic["p-norm"]
                    )
                )

        final_latents.append(latent_in)

        return final_latents

    def cal_loss(self, im_dict, latent_in):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic["p-norm"] = p_norm_loss
        loss += p_norm_loss

        return loss, loss_dic

    def save_results(self, ref_name, gen_im, latent_in, output_dir):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        os.makedirs(output_dir, exist_ok=True)

        latent_path = os.path.join(output_dir, f"{ref_name[0]}.npy")
        image_path = os.path.join(output_dir, f"{ref_name[0]}.png")

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def save_intermediate_results(self, ref_name, gen_im, latent_in, step, output_dir):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        intermediate_folder = os.path.join(output_dir, ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f"{ref_name[0]}_{step:04}.npy")
        image_path = os.path.join(intermediate_folder, f"{ref_name[0]}_{step:04}.png")

        save_im.save(image_path)
        np.save(latent_path, save_latent)

    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
