import numpy as np
import glob
import clip
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from typing import Union, List, Dict
from core.utils.common import mixing_noise, load_clip
from tqdm.auto import tqdm

import core.utils.fid as fid


def get_tril_elements_mask(linear_size):
    mask = np.zeros((linear_size, linear_size), dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    np.fill_diagonal(mask, False)
    return mask


class Evaluator:
    def __init__(self, config, image_based=False):
        self.config = config
        self.device = config.training.device
        self.image_based = image_based

        self.models = {
            visual_encoder: load_clip(visual_encoder, device=self.config.training.device)
            for visual_encoder in self.config.evaluation.vision_models
        }
        
    @torch.no_grad()
    def _encode_text(
        self, clip_model: nn.Module, text: str, templates: List[str] = ("A {}",)
    ):
        tokens = clip.tokenize(t.format(text) for t in templates).to(self.device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def _encode_target_image(
        self, clip_model: nn.Module, preprocess, target_image: str
    ):
        preprocessed = preprocess(Image.open(target_image)).unsqueeze(0).to(self.device)
        target_encoding = clip_model.encode_image(preprocessed)
        target_encoding /= target_encoding.clone().norm(dim=-1, keepdim=True)

        return target_encoding

    @torch.no_grad()
    def _encode_image(self, clip_model: nn.Module, preprocess, imgs: torch.Tensor):
        images = preprocess(imgs).to(self.device)
        image_features = clip_model.encode_image(images).detach()
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def _mean_cosine_sim(self, imgs_encoded: torch.Tensor, mean_vector: torch.Tensor):
        return (imgs_encoded.unsqueeze(1) * mean_vector).sum(dim=-1).mean().item()

    def _std_cosine_sim(self, imgs_encoded: torch.Tensor, mean_vector: torch.Tensor):
        return nn.CosineSimilarity()(imgs_encoded, mean_vector).std().item()

    def _diversity_from_embeddings_pairwise_cosines(self, imgs_encoded: torch.Tensor):
        data = (imgs_encoded @ imgs_encoded.T).cpu().numpy()
        mask = get_tril_elements_mask(data.shape[0])
        return np.mean(1 - data[mask])

    @torch.no_grad()
    def _generate_clip_data(
        self,
        clip_model,
        preprocess,
        source_generator,
        trainable,
        mapper_input=None,
    ):
        answer = []

        for idx in tqdm(range(self.config.evaluation.data_size // self.config.evaluation.batch_size)):

            sample_z = mixing_noise(
                self.config.evaluation.batch_size,
                512,
                self.config.training.mixing_noise,
                self.config.training.device
            )

            if self.config.training.patch_key == "original":
                imgs, _ = trainable(sample_z, input_is_latent=False)
            elif "mapper_config" in self.config.training:
                imgs, _ = source_generator(
                    sample_z,
                    offsets=trainable(mapper_input),
                    input_is_latent=False,
                )
            else:
                imgs, _ = source_generator(
                    sample_z, offsets=trainable(), input_is_latent=False
                )

            image_features = self._encode_image(clip_model, preprocess, imgs).detach()
            answer.append(image_features)

        return torch.cat(answer, dim=0)
    
    def get_metrics(
        self,
        source_generator,
        trainable, 
        target,
        mapper_inputs=None,
    ):

        metrics = {}
        trainable.eval()
        source_generator.eval()

        if isinstance(target, str):
            target = [target]

        for key, (clip_model, preprocess) in self.models.items():
            for target_class in target:
                if mapper_inputs is not None:
                    mapper_input = mapper_inputs[target_class]
                else:
                    mapper_input = None
                
                domain_mean_vector = self._encode_text(clip_model, target_class).unsqueeze(0)
                imgs_encoded = self._generate_clip_data(
                    clip_model,
                    preprocess,
                    source_generator,
                    trainable,
                    mapper_input,
                )
                domain_mean_vector = self._encode_text(clip_model, target_class)
                cls_description = "_".join(target_class.lower().split())

                key_quality = f"quality/{cls_description}/{key.replace('/', '-')}"
                key_diversity = f"diversity/{cls_description}/{key.replace('/', '-')}"

                metrics[key_quality] = self._mean_cosine_sim(
                    imgs_encoded, domain_mean_vector
                )

                metrics[
                    key_diversity
                ] = self._diversity_from_embeddings_pairwise_cosines(imgs_encoded)

        return metrics


class MTGEvaluator(Evaluator):
    @torch.no_grad()
    def _generate_data_for_fid(
        self,
        source_generator,
        trainable,
        config,
        latents=None,
        mean_latent=None,
    ):
        imgs, c_imgs = [], []

        for idx in range(config.evaluation.data_size // config.evaluation.batch_size):
            sample_z = mixing_noise(
                24, 512, config.training.mixing_noise, config.training.device
            )
            w_styles = source_generator.style(sample_z)
            if config.training.patch_key == "original":
                sampled_images, _ = trainable(
                    sample_z, truncation=config.logging.truncation
                )
            else:
                sampled_images, _ = source_generator(
                    sample_z,
                    weights_deltas=trainable(),
                    truncation=config.logging.truncation,
                )

            tmp_latents = 0.5 * (w_styles[0] - mean_latent) + mean_latent
            tmp_latents = tmp_latents.unsqueeze(1).repeat(1, 18, 1)
            tmp_latents[:, 7:, :] = latents[:, 7:, :]

            if config.training.patch_key == "original":
                color_img = trainable([tmp_latents], input_is_latent=True)[0]
            else:
                color_img = source_generator(
                    [tmp_latents],
                    weights_deltas=trainable(),
                    input_is_latent=True,
                )[0]

            sampled_images = transforms.Resize(256)(sampled_images)
            color_img = transforms.Resize(256)(color_img)

            imgs.append(sampled_images.detach())
            c_imgs.append(color_img.detach())

        return (
            torch.cat(imgs, dim=0),
            torch.cat(c_imgs, dim=0),
        )

    @torch.no_grad()
    def _generate_data(
        self,
        clip_model,
        preprocess,
        source_generator,
        trainable,
        config,
        embedding_to_mapper,
        latents=None,
        mean_latent=None,
    ):
        img_features = []
        color_features = []

        for idx in range(config.evaluation.data_size // config.evaluation.batch_size):
            sample_z = mixing_noise(
                24, 512, config.training.mixing_noise, config.training.device
            )
            w_styles = source_generator.style(sample_z)
            if config.training.patch_key == "original":
                sampled_images, _ = trainable(
                    sample_z, truncation=config.logging.truncation
                )

            else:
                sampled_images, _ = source_generator(
                    sample_z,
                    weights_deltas=trainable(),
                    truncation=config.logging.truncation,
                )

            tmp_latents = 0.5 * (w_styles[0] - mean_latent) + mean_latent
            tmp_latents = tmp_latents.unsqueeze(1).repeat(1, 18, 1)
            tmp_latents[:, 7:, :] = latents[:, 7:, :]

            if config.training.patch_key == "original":
                color_img = trainable([tmp_latents], input_is_latent=True)[0]
            else:
                color_img = source_generator(
                    [tmp_latents],
                    weights_deltas=trainable(),
                    input_is_latent=True,
                )[0]

            sampled_images = sampled_images.detach()
            color_img = color_img.detach()
            image_features = self._encode_image(clip_model, preprocess, sampled_images)
            c_features = self._encode_image(clip_model, preprocess, color_img)

            img_features.append(image_features)
            color_features.append(c_features)

        return torch.cat(img_features, dim=0), torch.cat(color_features, dim=0)

    def get_metrics(
        self,
        source_generator: nn.Module,
        trainable: nn.Module,
        target: Union[List[str], str],
        mapper_input_embeddings: Dict = None,
        latents=None,
        mean_latent=None,
    ):

        metrics = {}
        trainable.eval()
        source_generator.eval()

        if isinstance(target, str):
            target = [target]

        for key, (clip_model, preprocess) in self.models.items():
            for target_class in target:
                if mapper_input_embeddings is not None:
                    mapper_input = mapper_input_embeddings[target_class]
                else:
                    mapper_input = None

                imgs_encoded, sm_encoded = self._generate_data(
                    clip_model,
                    preprocess,
                    source_generator,
                    trainable,
                    self.config,
                    mapper_input,
                    latents,
                    mean_latent,
                )

                if (target_class, key) not in self.text_embeddings:
                    if self.image_based:
                        domain_mean_vector = self._encode_target_image(
                            clip_model, self.clip_preprocesses[key], target_class
                        )
                    else:
                        domain_mean_vector = self._encode_text(clip_model, target_class)
                    self.text_embeddings[(target_class, key)] = domain_mean_vector

                cls_description = "_".join(target_class.lower().split())

                key_quality = f"quality/{cls_description}/{key.replace('/', '-')}"
                key_diversity = f"diversity/{cls_description}/{key.replace('/', '-')}"

                for pref, imgs in zip(
                    ["", "sm_"], [imgs_encoded, sm_encoded]
                ):
                    metrics[pref + key_quality] = self._mean_cosine_sim(
                        imgs, self.text_embeddings[(target_class, key)]
                    )

                    metrics[
                        pref + key_diversity
                    ] = self._diversity_from_embeddings_pairwise_cosines(imgs)

        if self.config.evaluation.fid:
            imgs_ref = []
            transform = transforms.ToTensor()
            for p in glob.glob(self.config.evaluation.fid_ref + "*"):
                img = Image.open(p).convert("RGB")
                img = transform(img)
                imgs_ref.append(img)
            imgs_ref = torch.stack(imgs_ref)

            imgs, wc_imgs, c_imgs = self._generate_data_for_fid(
                source_generator, trainable, self.config, latents, mean_latent
            )
            for name, imgs in zip(["imgs", "wc", "c"], [imgs, wc_imgs, c_imgs]):
                metrics[f"fid/{name}"] = fid.calculate_fid(
                    imgs_ref,
                    imgs,
                    self.config.evaluation.batch_size,
                    self.device,
                    2048,
                )

        return metrics

