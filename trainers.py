import os
import wandb

import torch
import torch.nn as nn
import numpy as np
import clip
import torchvision.transforms as transforms
import torch.distributions as dis
import typing as tp

from pathlib import Path

from core.utils.text_templates import imagenet_templates
from core.utils.train_log import StreamingMeans, TimeLog, Timer
from core.utils.loggers import LoggingManager
from core.utils.class_registry import ClassRegistry
from core.utils.common import (
    mixing_noise, validate_device, compose_text_with_templates, load_clip,
    read_domain_list, read_style_images_list, determine_opt_layers,
    get_stylegan_conv_dimensions, DataParallelPassthrough, get_trainable_model_state
)

from core.loss import DirectLoss
from core.utils.math_utils import (
    resample_single_vector, resample_batch_vectors, convex_hull,
    resample_batch_templated_embeddings, convex_hull_small
)

from core.utils.image_utils import BicubicDownSample, t2im, construct_paper_image_grid, crop_augmentation
from core.parametrizations import BaseParametrization
from core.mappers import mapper_registry
from core.utils.II2S import II2S
from core.uda_models import uda_models
from core.dataset import ImagesDataset

trainer_registry = ClassRegistry()


class BaseDomainAdaptationTrainer:
    def __init__(self, config):
        # common
        self.config = config
        self.trainable = None
        self.source_generator = None

        self.current_step = 0
        self.optimizer = None
        self.loss_function = None
        self.batch_generators = None

        self.zs_for_logging = None

        self.reference_embeddings = {}

        # processed in multiple_domain trainer
        self.domain_embeddings = None
        self.desc_to_embeddings = None

        self.global_metrics = {}

    def _setup_base(self):
        self._setup_device()
        self._setup_logger()
        self._setup_batch_generators()
        self._setup_source_generator()
        self._setup_loss()
        self._initial_logging()

    def _setup_device(self):
        chosen_device = self.config.training["device"].lower()
        device = validate_device(chosen_device)
        self.device = torch.device(device)

    def _setup_source_generator(self):
        self.source_generator = uda_models[self.config.training.generator](
            **self.config.generator_args[self.config.training.generator]
        )
        self.source_generator.patch_layers(self.config.training.patch_key)
        self.source_generator.freeze_layers()
        self.source_generator.to(self.device)

    def _setup_loss(self):
        self.loss_function = DirectLoss(self.config.optimization_setup)

    def _setup_logger(self):
        self.logger = LoggingManager(self.config)

    def _setup_batch_generators(self):
        self.batch_generators = {}

        for visual_encoder in self.config.optimization_setup.visual_encoders:
            self.batch_generators[visual_encoder] = (
                load_clip(visual_encoder, device=self.config.training.device)
            )

        self.reference_embeddings = {k: {} for k in self.batch_generators}

    @torch.no_grad()
    def _initial_logging(self):
        self.zs_for_logging = [
            mixing_noise(16, 512, 0, self.config.training.device)
            for _ in range(self.config.logging.num_grid_outputs)
        ]

        for idx, z in enumerate(self.zs_for_logging):
            images = self.forward_source(z)
            self.logger.log_images(0, {f"src_domain_grids/{idx}": construct_paper_image_grid(images)})
    
    def _setup_optimizer(self):
        if self.config.training.patch_key == "original":
            g_reg_every = self.config.optimization_setup.g_reg_every
            lr = self.config.optimization_setup.optimizer.lr

            g_reg_ratio = g_reg_every / (g_reg_every + 1)
            betas = self.config.optimization_setup.optimizer.betas

            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(),
                lr=lr * g_reg_ratio,
                betas=(betas[0] ** g_reg_ratio, betas[1] ** g_reg_ratio),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.trainable.parameters(), **self.config.optimization_setup.optimizer
            )

    # @classmethod
    # def from_ckpt(cls, ckpt_path):
    #     m = cls(ckpt['config'])
    #     m._setup_base()
    #     return m

    def start_from_checkpoint(self):
        step = 0
        if self.config.checkpointing.start_from:
            state_dict = torch.load(self.config.checkpointing.start_from, map_location='cpu')
            step = state_dict['step']
            self.trainable.load_state_dict(state_dict['trainable'])
            self.optimizer.load_state_dict(state_dict['trainable_optimizer'])
            print('starting from step {}'.format(step))
        # TODO: python main.py --ckpt_path ./.... -> Trainer.from_ckpt()
        return step

    def get_checkpoint(self):
        state_dict = {
            "step": self.current_step,
            "trainable": self.trainable.state_dict(),
            "trainable_optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        return state_dict

    # TODO: refactor checkpoint
    def make_checkpoint(self):
        if not self.config.checkpointing.is_on:
            return

        ckpt = self.get_checkpoint()
        torch.save(ckpt, os.path.join(self.logger.checkpoint_dir, "checkpoint.pt"))

    def save_models(self):
        models_dict = get_trainable_model_state(
            self.config, self.trainable.state_dict()
        )
        
        models_dict.update(self.ckpt_info())
        torch.save(models_dict, str(
            Path(self.logger.models_dir) / f"models_{self.current_step}.pt"
        ))
    
    def ckpt_info(self):
        return {}
    
    def all_to_device(self, device):
        self.source_generator.to(device)
        self.trainable.to(device)
        self.loss_function.to(device)

    def train_loop(self):
        self.all_to_device(self.device)
        
        training_time_log = TimeLog(
            self.logger, self.config.training.iter_num + 1, event="training"
        )

        recovered_step = self.start_from_checkpoint()
        iter_info = StreamingMeans()

        for self.current_step in range(recovered_step, self.config.training.iter_num + 1, 1):
            with Timer(iter_info, "train_iter"):
                self.train_step(iter_info)

            if self.current_step % self.config.checkpointing.step_backup == 0:
                self.make_checkpoint()

            if (self.current_step + 1) % self.config.exp.step_save == 0:
                self.save_models()

            if self.current_step % self.config.logging.log_images == 0:
                with Timer(iter_info, "log_images"):
                    self.log_images()

            if self.current_step % self.config.logging.log_every == 0:
                self.logger.log_values(
                    self.current_step, self.config.training.iter_num, iter_info
                )
                iter_info.clear()
                training_time_log.now(self.current_step)

        training_time_log.end()
        wandb.finish()

    @torch.no_grad()
    def encode_text(self, model, text, templates):
        text = compose_text_with_templates(text, templates=templates)
        tokens = clip.tokenize(text).to(self.config.training.device)
        text_features = model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def clip_encode_image(self, model, image, preprocess):
        image_features = model.encode_image(preprocess(image))
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features
    
    def partial_trainable_model_freeze(self):
        if not hasattr(self.config.training, 'auto_layer_iters'):
            return
        
        if self.config.training.auto_layer_iters == 0:
            return

        train_layers = determine_opt_layers(
            self.source_generator,
            self.trainable,
            self.batch_generators['ViT-B/32'][0],
            self.config,
            self.config.training.target_class,
            self.config.training.auto_layer_iters,
            self.config.training.auto_layer_batch,
            self.config.training.auto_layer_k,
            device=self.device,
        )

        if not isinstance(train_layers, list):
            train_layers = [train_layers]

        self.trainable.freeze_layers()
        self.trainable.unfreeze_layers(train_layers)

    def train_step(self, iter_info):
        self.trainable.train()
        sample_z = mixing_noise(
            self.config.training.batch_size,
            512,
            self.config.training.mixing_noise,
            self.config.training.device,
        )

        self.partial_trainable_model_freeze()

        batch = self.calc_batch(sample_z)
        losses = self.loss_function(batch)
        
        self.optimizer.zero_grad()
        losses["total"].backward(retain_graph=True)
        self.optimizer.step()

        iter_info.update({f"losses/{k}": v for k, v in losses.items()})

    def forward_trainable(self, latents, *args, **kwargs):
        raise NotImplementedError()

    @torch.no_grad()
    def forward_source(self, latents, **kwargs) -> torch.Tensor:
        sampled_images, _ = self.source_generator(latents, **kwargs)
        return sampled_images.detach()

    def calc_batch(self, sample_z):
        raise NotImplementedError()
    
    @torch.no_grad()
    def log_images(self):
        raise NotImplementedError()

    def to_multi_gpu(self):
        self.source_generator = DataParallelPassthrough(self.source_generator, device_ids=self.config.exp.device_ids)
        self.trainable = DataParallelPassthrough(self.trainable, device_ids=self.config.exp.device_ids)

    def invert_image_ii2s(self, image_info, ii2s):
        image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(self.device)
        image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(self.device)

        lam = str(int(ii2s.opts.p_norm_lambda * 1000))
        name = Path(image_info['image_name']).stem + f"_{lam}.npy"
        current_latents_path = self.logger.cached_latents_local_path / name

        if current_latents_path.exists():
            latents = np.load(str(current_latents_path))
            latents = torch.from_numpy(latents).to(self.config.training.device)
        else:
            latents, = ii2s.invert_image(
                image_full_res,
                image_resized
            )

            print(f'''
            latents for {image_info['image_name']} cached in 
            {str(current_latents_path.resolve())}
            ''')

            np.save(str(current_latents_path), latents.detach().cpu().numpy())

        return latents


class SingleDomainAdaptationTrainer(BaseDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _setup_trainable(self):
        if self.config.training.patch_key == 'original':
            self.trainable = uda_models[self.config.training.generator](
                **self.config.generator_args[self.config.training.generator]
            )
            trainable_layers = list(self.trainable.get_training_layers(
                phase=self.config.training.phase
            ))
            self.trainable.freeze_layers()
            self.trainable.unfreeze_layers(trainable_layers)
        else:
            self.trainable = BaseParametrization(
                self.config.training.patch_key,
                get_stylegan_conv_dimensions(self.source_generator.generator.size),
            )

        self.trainable.to(self.device)
    
    def forward_trainable(self, latents, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if self.config.training.patch_key == "original":
            sampled_images, _ = self.trainable(
                latents, **kwargs
            )
            offsets = None
        else:
            offsets = self.trainable()
            sampled_images, _ = self.source_generator(
                latents, offsets=offsets, **kwargs
            )

        return sampled_images, offsets
    
    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            sampled_images, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)
            images = construct_paper_image_grid(sampled_images)
            dict_to_log.update({
                f"trg_domain_grids/{self.config.training.target_class}/{idx}": images
            })

        self.logger.log_images(self.current_step, dict_to_log)
        
    
@trainer_registry.add_to_registry("td_single")
class TextDrivenSingleDomainAdaptationTrainer(SingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def ckpt_info(self):
        return {
            'da_type': 'td',
        }
    
    def setup(self):
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()
        self._setup_text_embeddings()

    def _setup_text_embeddings(self):
        for visual_encoder, (model, preprocess) in self.batch_generators.items():
            self.reference_embeddings[visual_encoder][self.config.training.source_class] = self.encode_text(
                model, self.config.training.source_class, imagenet_templates
            )
            self.reference_embeddings[visual_encoder][self.config.training.target_class] = self.encode_text(
                model, self.config.training.target_class, imagenet_templates
            )

    def calc_batch(self, sample_z):
        clip_data = {
            k: {} for k in self.batch_generators
        }
        
        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)
                
        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
            
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            
            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.target_class].unsqueeze(0)
                ),
                'src_domain_emb': (
                    self.reference_embeddings[visual_encoder_key][self.config.training.source_class].unsqueeze(0)
                )
            })
        
        return {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }
    

@trainer_registry.add_to_registry("im2im_single")
class Image2ImageSingleDomainAdaptationTrainer(SingleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.style_image_latents = None
        self.style_image_full_res = None
        self.style_image_resized = None
        self.style_image_inverted_A = None
    
    def setup(self):
        self._setup_base()
        self._setup_trainable()
        self._setup_optimizer()

        self._setup_style_image()
        self._log_target_images()
        self._setup_src_embeddings()
    
    def ckpt_info(self):
        return {
            'da_type': 'im2im',
        }
    
    def _setup_src_embeddings(self):
        for visual_encoder, (model, preprocess) in self.batch_generators.items():
            self.reference_embeddings[visual_encoder][self.config.training.source_class] = self.encode_text(
                model, self.config.training.source_class, ['A {}']
            )

    def _setup_style_image(self):
        from core.style_embed_options import II2S_s_opts
        ii2s = II2S(II2S_s_opts)

        single_image_dataset = ImagesDataset(
            opts=II2S_s_opts,
            image_path=self.config.training.target_class,
            align_input=False
        )

        image_info = single_image_dataset[0]
        self.bicubic = BicubicDownSample(4)

        self.style_image_full_res = image_info['image_high_res_torch'].unsqueeze(0).to(self.device)
        self.style_image_resized = image_info['image_low_res_torch'].unsqueeze(0).to(self.device)

        self.style_image_latents = self.invert_image_ii2s(image_info, ii2s).detach().clone()
        self.style_image_inverted_A = self.forward_source([self.style_image_latents], input_is_latent=True)

    def _log_target_images(self):
        style_image_resized = t2im(self.style_image_resized.squeeze())
        st_im_inverted_A = t2im(self.style_image_inverted_A.squeeze())
        self.logger.log_images(
            0, {"style_image/orig": style_image_resized, "style_image/projected_A": st_im_inverted_A}
        )

    def calc_batch(self, sample_z):
        clip_data = {k: {} for k in self.batch_generators}

        frozen_img = self.forward_source(sample_z)
        trainable_img, offsets = self.forward_trainable(sample_z)
        style_image_inverted_B, _ = self.forward_trainable(
            [self.style_image_latents], input_is_latent=True
        )

        for visual_encoder_key, (model, preprocess) in self.batch_generators.items():
            
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            trg_domain_emb = self.clip_encode_image(model, self.style_image_full_res, preprocess)
            src_domain_emb = self.clip_encode_image(model, self.style_image_inverted_A, preprocess)
            # src_domain_emb = self.reference_embeddings[visual_encoder_key][self.config.training.source_class]
            st_inverted_B_emb = self.clip_encode_image(model, style_image_inverted_B, preprocess)
            st_orig_emb = self.clip_encode_image(model, self.style_image_full_res, preprocess)
            
            clip_data[visual_encoder_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': trg_domain_emb,
                'src_domain_emb': src_domain_emb,
                'st_inv_in_B_emb': st_inverted_B_emb,
                'st_orig_emb': st_orig_emb
            })

        rec_data = {
            'style_inverted_B_256x256': self.bicubic(style_image_inverted_B),
            'style_image_256x256': self.style_image_resized,
        }

        return {
            'clip_data': clip_data,
            'rec_data': rec_data,
            'offsets': offsets
        }

    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            sampled_imgs, _ = self.forward_trainable(z, truncation=self.config.logging.truncation)

            tmp_latents = w_styles[0].unsqueeze(1).repeat(1, 18, 1)
            gen_mean = self.source_generator.mean_latent.unsqueeze(1).repeat(1, 18, 1)
            style_mixing_latents = self.config.logging.truncation * (tmp_latents - gen_mean) + gen_mean
            style_mixing_latents[:, 7:, :] = self.style_image_latents[:, 7:, :]

            style_mixing_imgs, _ = self.forward_trainable(
                [style_mixing_latents], input_is_latent=True,
                truncation=1
            )

            sampled_imgs = construct_paper_image_grid(sampled_imgs)
            style_mixing_imgs = construct_paper_image_grid(style_mixing_imgs)

            dict_to_log.update({
                f"trg_domain_grids/{Path(self.config.training.target_class).stem}/{idx}": sampled_imgs,
                f"trg_domain_grids_sm/{Path(self.config.training.target_class).stem}/{idx}": style_mixing_imgs,

            })

        rec_img, _ = self.forward_trainable(
            [self.style_image_latents],
            input_is_latent=True,
        )
        rec_img = t2im(rec_img.squeeze())
        dict_to_log.update({"style_image/projected_B": rec_img})

        self.logger.log_images(self.current_step, dict_to_log)


class MultipleDomainAdaptationTrainer(BaseDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _setup_trainable(self):
        self.trainable = mapper_registry[
            self.config.mapper_config.mapper_type
        ](
            self.config.mapper_config,
            get_stylegan_conv_dimensions(self.source_generator.generator.size),
        )
        self.trainable.to(self.device)

    def forward_trainable(self, latents, embeddings, **kwargs) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        offsets = self.trainable(embeddings)
        sampled_images, _ = self.source_generator(
            latents, offsets=offsets, **kwargs
        )
        return sampled_images, offsets

    def _setup_mapper_encoder_model(self):
        self.mapper_encoder_type = 'ViT-B/32'
        self.mapper_encoder, self.mapper_preprocess = load_clip(self.mapper_encoder_type, self.config.training.device)


@trainer_registry.add_to_registry("im2im_multiple_base")
class IM2IMMultipleBaseTraienr(MultipleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def ckpt_info(self):
        return {
            'da_type': 'im2im',
        }
    
    def setup(self):
        self._setup_base()
        self._setup_mapper_encoder_model()
        self._setup_style_images()
        self._log_target_images()
        self._setup_trainable()
        self._setup_optimizer()

    def _setup_style_images(self):
        from core.style_embed_options import II2S_s_opts

        self.bicubic = BicubicDownSample(4)

        style_paths = read_style_images_list(self.config.training.train_styles)

        dataset = ImagesDataset(
            opts=II2S_s_opts,
            image_path=style_paths,
            align_input=False
        )

        ii2s = II2S(II2S_s_opts)

        self.train_styles = []
        self.name_to_info = {}
        self.mapper_embeddings = {}

        for image_info in dataset:
            self.train_styles.append(image_info['image_name'])
            latents = self.invert_image_ii2s(image_info, ii2s)
            style_image_inverted_A = self.forward_source([latents], input_is_latent=True)

            self.name_to_info[image_info['image_name']] = (
                latents.detach().clone(),
                image_info['image_high_res_torch'].unsqueeze(0).to(self.device),
                image_info['image_low_res_torch'].unsqueeze(0).to(self.device),
                style_image_inverted_A
            )

            self.mapper_embeddings[image_info['image_name']] = self.clip_encode_image(
                self.mapper_encoder,
                image_info['image_high_res_torch'].unsqueeze(0).to(self.device),
                self.mapper_preprocess
            ).float()

    def _log_target_images(self):
        log_data = {}
        for image_name in self.name_to_info:
            _, _, style_image_resized, style_image_inverted_A = self.name_to_info[image_name]

            style_image_resized = t2im(style_image_resized.squeeze())
            st_im_inverted_A = t2im(style_image_inverted_A.squeeze())

            log_data.update({
                f"style_image/orig/{image_name}": style_image_resized,
                f"style_image/projected_A/{image_name}": st_im_inverted_A
            })

        self.logger.log_images(
            0, log_data
        )
    
    def calc_batch(self, sample_z):
        clip_data = {
            k: {} for k in self.batch_generators
        }

        frozen_img = self.forward_source(sample_z)
        batch_styles = np.random.choice(
            self.train_styles, size=self.config.training.batch_size
        )

        st_lat, st_im_fullres, st_im_resized, st_im_inverted_A = zip(*[self.name_to_info[st] for st in batch_styles])
        st_lat = torch.cat(st_lat, dim=0)
        st_im_fullres = torch.cat(st_im_fullres, dim=0)
        st_im_resized = torch.cat(st_im_resized, dim=0)
        st_im_inverted_A = torch.cat(st_im_inverted_A, dim=0)

        trg_embeddings = torch.cat([self.mapper_embeddings[d] for d in batch_styles], dim=0)

        trainable_img, offsets = self.forward_trainable(sample_z, trg_embeddings)
        style_image_inverted_B, _ = self.forward_trainable([st_lat], trg_embeddings)

        for enc_key, (model, preprocess) in self.batch_generators.items():
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)
            trg_domain_embs = self.clip_encode_image(model, st_im_fullres, preprocess)
            src_domain_embs = self.clip_encode_image(model, st_im_inverted_A, preprocess)

            st_inverted_B_emb = self.clip_encode_image(model, style_image_inverted_B, preprocess)
            st_orig_emb = self.clip_encode_image(model, st_im_fullres, preprocess)

            clip_data[enc_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': trg_domain_embs,
                'src_domain_emb': src_domain_embs,
                'st_inv_in_B_emb': st_inverted_B_emb,
                'st_orig_emb': st_orig_emb
            })

        rec_data = {
            'style_inverted_B_256x256': self.bicubic(style_image_inverted_B),
            'style_image_256x256': st_im_resized,
        }

        return {
            'clip_data': clip_data,
            'rec_data': rec_data,
            'offsets': offsets
        }

    @torch.no_grad()
    def log_images(self):
        self.trainable.eval()
        dict_to_log = {}

        for idx, z in enumerate(self.zs_for_logging):
            w_styles = self.source_generator.style(z)
            for im_name in self.train_styles:
                sampled_imgs, offsets = self.forward_trainable(
                    z, self.mapper_embeddings[im_name],
                    truncation=self.config.logging.truncation
                )

                tmp_latents = w_styles[0].unsqueeze(1).repeat(1, 18, 1)
                gen_mean = self.source_generator.mean_latent.unsqueeze(1).repeat(1, 18, 1)
                style_mixing_latents = self.config.logging.truncation * (tmp_latents - gen_mean) + gen_mean
                st_latents, _, _, _ = self.name_to_info[im_name]
                style_mixing_latents[:, 7:, :] = st_latents[:, 7:, :]

                style_mixing_imgs, offsets = self.forward_trainable(
                    [style_mixing_latents],
                    self.mapper_embeddings[im_name],
                    input_is_latent=True
                )

                sampled_imgs = construct_paper_image_grid(sampled_imgs)
                style_mixing_imgs = construct_paper_image_grid(style_mixing_imgs)

                dict_to_log.update({
                    f"trg_domain_grids/{im_name}/{idx}": sampled_imgs,
                    f"trg_domain_grids_sm/{im_name}/{idx}": style_mixing_imgs,
                })

        for im_name in self.train_styles:
            st_latents, _, _, _ = self.name_to_info[im_name]
            rec_img, _ = self.forward_trainable(
                [st_latents],
                self.mapper_embeddings[im_name],
                input_is_latent=True,
            )
            rec_img = t2im(rec_img.squeeze())
            dict_to_log.update({f"style_image/projected_B/{im_name}": rec_img})

        self.logger.log_images(self.current_step, dict_to_log)


@trainer_registry.add_to_registry("td_multiple_base")
class TextDrivenMultiTrainer(MultipleDomainAdaptationTrainer):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        self._setup_base()
        self._setup_mapper_encoder_model()
        self._setup_domain_embeddings()
        self._setup_trainable()
        self._setup_optimizer()
    
    def ckpt_info(self):
        return {
            'da_type': 'td',
        }
    
    @torch.no_grad()
    def _read_domains(self, path, model, templates=imagenet_templates):
        """
        process txt file with format
            'Target Class - Source Class'
            'Target Class - Source Class'

            e.g.

            'Anime Painting - Photo'
            'Cubism Painting - Photo'

        Parameters
        ----------
        path : str
            The file location of the target-source domain pairs
        model : nn.Module
            clip encoder that is used for text encoding
        templates : list[str]
            templates for text augmentation

        Returns
        -------
        desc_to_embeddings : mapping[str, torch.Tensor]
            mapping, text description to corresponding torch tensor of encoded text

        trg_to_src : mapping[str, str]
            mapping of target domain to corresponding source one
        """
        domain_list = read_domain_list(path)
        desc_to_embeddings = {}
        target_to_source_mapping = {}

        for target_domain, source_domain in domain_list:
            if target_domain not in desc_to_embeddings:
                text_features = self.encode_text(model, target_domain, templates)
                text_features = text_features.to(self.config.training.device).float()
                target_to_source_mapping[target_domain] = source_domain
                desc_to_embeddings[target_domain] = text_features.to(self.config.training.device).float()

            if source_domain not in desc_to_embeddings:
                text_features = self.encode_text(model, source_domain, templates)
                text_features = text_features.to(self.config.training.device).float()

                desc_to_embeddings[source_domain] = text_features.to(self.config.training.device).float()

        return desc_to_embeddings, target_to_source_mapping

    def _setup_domain_embeddings(self):
        self.reference_embeddings = {
            k: self._read_domains(self.config.training.train_domain_list, m)[0] for k, (m, p) in self.batch_generators.items()
        }

        self.mapper_train_inputs, self.train_target_to_source_mapping = \
            self._read_domains(self.config.training.train_domain_list, self.mapper_encoder, ("{}", ))
        self.target_domains_train = list(self.train_target_to_source_mapping.keys())

        self.mapper_test_inputs, self.test_target_to_source_mapping = \
            self._read_domains(self.config.training.test_domain_list, self.mapper_encoder, ("{}", ))
        self.target_domains_test = list(self.test_target_to_source_mapping.keys())

    def calc_batch(self, sample_z):
        clip_data = {
            k: {} for k in self.batch_generators
        }

        frozen_img = self.forward_source(sample_z)

        batch_descs = np.random.choice(
            self.target_domains_train, size=self.config.training.batch_size
        )

        trg_embeddings = torch.cat([self.mapper_train_inputs[d] for d in batch_descs], dim=0)
        trainable_img, offsets = self.forward_trainable(sample_z, trg_embeddings)

        for enc_key, (model, preprocess) in self.batch_generators.items():
            trg_encoded = self.clip_encode_image(model, trainable_img, preprocess)
            src_encoded = self.clip_encode_image(model, frozen_img, preprocess)

            trg_domain_embs = torch.stack(
                [self.reference_embeddings[enc_key][t] for t in batch_descs]
            )

            src_domain_embs = torch.stack(
                [self.reference_embeddings[enc_key][self.train_target_to_source_mapping[t]] for t in batch_descs]
            )

            clip_data[enc_key].update({
                'trg_encoded': trg_encoded,
                'src_encoded': src_encoded,
                'trg_domain_emb': trg_domain_embs,
                'src_domain_emb': src_domain_embs
            })

        return {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }

    def _log_images_multi(self, descriptions, event_log=''):
        dict_to_log = {}
        for idx, z in enumerate(self.zs_for_logging):
            for test_description in descriptions:
                images, _ = self.forward_trainable(
                    z, self.mapper_test_inputs[test_description],
                    truncation=self.config.logging.truncation
                )

                dict_to_log.update({
                    f"test_domain_grids/{test_description}/{idx}": construct_paper_image_grid(images)
                })

        self.logger.log_images(self.current_step, dict_to_log)

    @torch.no_grad()
    def log_images(self, event_log=''):
        self.trainable.eval()
        self._log_images_multi(self.target_domains_test)

        
@trainer_registry.add_to_registry('td_multiple_resample_and_convex')
class ParametrizationDomainAdaptationTrainer(TextDrivenMultiTrainer):
    def calc_alphas(self, initial, needed):
        m = dis.dirichlet.Dirichlet(torch.ones(needed, initial, device=self.config.training.device) / initial * 2)
        a = m.sample()
        return a

    def calc_batch(self, sample_z):
        clip_data = {
            k: {} for k in self.batch_generators
        }

        frozen_img = self.forward_source(sample_z)
        batch_descs = np.random.choice(self.target_domains_train, size=self.config.training.batch_size)
        src_batch_descs = [self.train_target_to_source_mapping[dom] for dom in batch_descs]
        
        alphas = self.calc_alphas(self.config.training.batch_size, self.config.training.batch_size)

        mapper_input = torch.cat([self.mapper_train_inputs[dom] for dom in batch_descs], dim=0)
        
        if self.config.convex_hull.do:
            mapper_input = convex_hull_small(mapper_input, alphas)

        trainable_img, offsets = self.forward_trainable(sample_z, mapper_input)

        for enc_key, (model, preprocess) in self.batch_generators.items():
            if self.config.resample.do and enc_key != self.mapper_encoder_type:
                clip_data.pop(enc_key)
                continue
            
            src_emb = torch.stack([self.reference_embeddings[enc_key][dom] for dom in src_batch_descs], dim=0)
            trg_emb = torch.stack([self.reference_embeddings[enc_key][dom] for dom in batch_descs], dim=0)
            
            if self.config.convex_hull.do:
                src_emb = convex_hull(src_emb, alphas)
                trg_emb = convex_hull(trg_emb, alphas)
            
            if self.config.resample.do:
                trg_emb = resample_batch_templated_embeddings(trg_emb, self.config.resample.divergence)
            
            clip_data[enc_key]['src_domain_emb'] = src_emb
            clip_data[enc_key]['trg_domain_emb'] = trg_emb
            clip_data[enc_key]['src_encoded'] = model.encode_image(
                preprocess(crop_augmentation(frozen_img))
            )
            clip_data[enc_key]['trg_encoded'] = model.encode_image(
                preprocess(crop_augmentation(trainable_img))
            )

        return {
            'clip_data': clip_data,
            'rec_data': None,
            'offsets': offsets
        }
