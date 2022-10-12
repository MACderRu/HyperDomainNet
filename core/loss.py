import torch
import torch.nn as nn
import typing as tp

from collections import defaultdict
from core import lpips

from core.utils.loss_utils import (
    cosine_loss,
    mse_loss,
    get_tril_elements_mask
)
from core.utils.class_registry import ClassRegistry


clip_loss_registry = ClassRegistry()
rec_loss_registry = ClassRegistry()
reg_loss_registry = ClassRegistry()


class LossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(LossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, "l2"], [opt.percept_lambda, "percep"]]
        self.l2 = torch.nn.MSELoss()
        if opt.device == "cuda":
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()
        # self.percept = VGGLoss()

    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)

    def _loss_lpips(self, gen_im, ref_im, **kwargs):
        return self.percept(gen_im, ref_im).sum()

    def forward(self, ref_im_H, ref_im_L, gen_im_H, gen_im_L):
        loss = 0
        loss_fun_dict = {
            "l2": self._loss_l2,
            "percep": self._loss_lpips,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == "l2":
                var_dict = {
                    "gen_im": gen_im_H,
                    "ref_im": ref_im_H,
                }
            elif loss_type == "percep":
                var_dict = {
                    "gen_im": gen_im_L,
                    "ref_im": ref_im_L,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight * tmp_loss
        return loss, losses


@reg_loss_registry.add_to_registry("offsets_l2")
def l2_offsets(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        layer_deltas = sum([v for v in conv_inputs.values()])
        loss += torch.pow(layer_deltas, 2).sum() / torch.numel(layer_deltas)

    return loss


@reg_loss_registry.add_to_registry("affine_l2")
def cout_affine_l2_loss(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        val = (torch.pow(conv_inputs['gamma'] - 1, 2) + torch.pow(conv_inputs['beta'], 2)).sum()
        loss += val / torch.numel(conv_inputs['gamma'])
    return  loss


@reg_loss_registry.add_to_registry("offsets_l1")
def l1_offsets(
    offsets: tp.Dict[str, tp.Dict[str, torch.Tensor]]
):
    loss = 0.
    for conv_key, conv_inputs in offsets.items():
        layer_deltas = sum([v for v in conv_inputs.values()])
        loss += torch.abs(layer_deltas).sum() / torch.numel(layer_deltas)

    return loss


@clip_loss_registry.add_to_registry("global")
def global_loss(
    trg_encoded: torch.Tensor, src_encoded: torch.Tensor,
    trg_domain_emb: torch.Tensor, src_domain_emb: torch.Tensor
) -> torch.Tensor:

    return cosine_loss(trg_encoded, trg_domain_emb).mean()

# batch /
#     clip_data /
#         ViT-B/32 /
#             src_encoded
#             (src_emb_domain..)


@clip_loss_registry.add_to_registry("direction")
def direction_loss(
    trg_encoded: torch.Tensor, src_encoded: torch.Tensor,
    trg_domain_emb: torch.Tensor, src_domain_emb: torch.Tensor
) -> torch.Tensor:

    edit_im_direction = trg_encoded - src_encoded
    edit_domain_direction = trg_domain_emb - src_domain_emb
        
    if trg_domain_emb.ndim == 3:
        edit_domain_direction = edit_domain_direction.mean(axis=1)
        
    return cosine_loss(edit_im_direction, edit_domain_direction).mean()


@clip_loss_registry.add_to_registry("indomain")
def indomain_loss(
    trg_encoded: torch.Tensor, src_encoded: torch.Tensor,
    trg_domain_emb: torch.Tensor, src_domain_emb: torch.Tensor
) -> torch.Tensor:
            
    src_cosines = src_encoded @ src_encoded.T
    trg_cosines = trg_encoded @ trg_encoded.T
    mask = torch.from_numpy(get_tril_elements_mask(src_encoded.size(0)))
    
    src_cosines = src_cosines[mask]
    trg_cosines = trg_cosines[mask]

    loss = torch.sum((src_cosines - trg_cosines) ** 2) / src_encoded.size(0) / (src_encoded.size(0) - 1) * 2

    return loss


@clip_loss_registry.add_to_registry("tt_direction")
def target_target_direction(
    trg_encoded: torch.Tensor, src_encoded: torch.Tensor,
    trg_domain_emb: torch.Tensor, src_domain_emb: torch.Tensor
) -> torch.Tensor:

    mask = torch.from_numpy(get_tril_elements_mask(trg_encoded.size(0)))
    
    deltas_text = (trg_domain_emb.unsqueeze(0) - trg_domain_emb.unsqueeze(1))[mask]
    deltas_img = (trg_encoded.unsqueeze(0) - trg_encoded.unsqueeze(1))[mask]
    
    zero_mask_im = torch.isclose(deltas_img.sum(dim=1).float(), torch.tensor(0.).float())
    zero_mask_t = torch.isclose(deltas_text.sum(dim=1).sum(dim=1).float(), torch.tensor(0.).float())

    non_zeromask = ~(zero_mask_im & zero_mask_t)
        
    if trg_domain_emb.ndim == 3:
        deltas_text = deltas_text.mean(dim=1)
    
    res_loss = cosine_loss(deltas_img[non_zeromask].float(), deltas_text[non_zeromask].float())
    
    return res_loss.mean()


@clip_loss_registry.add_to_registry('clip_within')
def clip_within(
    trg_encoded: torch.Tensor,
    src_encoded: torch.Tensor,
    style_image_trg_encoded: torch.Tensor,
    style_image_src_encoded: torch.Tensor,
):
    
    trg_direction = trg_encoded - style_image_trg_encoded
    src_direction = src_encoded - style_image_src_encoded

    return cosine_loss(trg_direction, src_direction).mean()


@rec_loss_registry.add_to_registry('clip_ref')
def clip_ref(
    batch: tp.Dict[str, tp.Dict]
):
    loss = 0.
    
    for visual_encoder_key, clip_batch in batch['clip_data'].items():
        log_vienc_key = visual_encoder_key.replace('/', '-')
        st_inv_in_B_emb = clip_batch['st_inv_in_B_emb']
        st_orig_emb = clip_batch['st_orig_emb']
        
        loss += cosine_loss(st_inv_in_B_emb, st_orig_emb).mean()

    return loss / len(batch['clip_data'])


@rec_loss_registry.add_to_registry('l2_rec')
def l2_rec(
    batch: tp.Dict[str, torch.Tensor]
):
    return mse_loss(
        batch['rec_data']['style_inverted_B_256x256'],
        batch['rec_data']['style_image_256x256']
    ).mean()


@rec_loss_registry.add_to_registry('lpips_rec')
def lpips_rec(
    batch: tp.Dict[str, torch.Tensor]
):
    if hasattr(lpips_rec, '_model'):
        lpips_rec._model = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    
    return lpips_rec._model(
        batch['rec_data']['style_inverted_B_256x256'],
        batch['rec_data']['style_image_256x256']
    ).mean()


class BaseLoss:
    loss_registry = []

    def __init__(self, loss_funcs, loss_coefs):
        self.funcs, self.coefs = [], []
        for func, coef in zip(loss_funcs, loss_coefs):
            if func not in self.loss_registry:
                continue
            
            self.funcs.append(func)
            self.coefs.append(coef)

    def __call__(self, batch):
        raise NotImplementedError()


class CLIPLoss(BaseLoss):
    loss_registry = clip_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)

        for loss, coef in zip(self.funcs, self.coefs):
            for visual_encoder_key, clip_batch in batch['clip_data'].items():
                log_vienc_key = visual_encoder_key.replace('/', '-')
                src_encoded = clip_batch['src_encoded']
                src_domain_emb = clip_batch['src_domain_emb']
                trg_encoded = clip_batch['trg_encoded']
                trg_domain_emb = clip_batch['trg_domain_emb']
                
                losses[f'{loss}_{log_vienc_key}'] = coef * self.loss_registry[loss](
                    trg_encoded, src_encoded, trg_domain_emb, src_domain_emb
                )

        return losses


class RecLoss(BaseLoss):
    loss_registry = rec_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)

        for loss, coef in zip(self.funcs, self.coefs):
            losses[f'{loss}'] = coef * self.loss_registry[loss](
                batch
            )

        return losses


class RegLoss(BaseLoss):
    loss_registry = reg_loss_registry

    def __call__(self, batch):
        losses = defaultdict(float)
        
        for loss, coef in zip(self.funcs, self.coefs):
            losses[f'{loss}'] = coef * self.loss_registry[loss](
                batch['offsets']
            )

        return losses


class DirectLoss(nn.Module):
    def __init__(self, loss_config):
        super().__init__()
        self.config = loss_config
        self.loss_funcs = loss_config.loss_funcs
        self.loss_coefs = loss_config.loss_coefs

        for key, c in zip(
            ['clip', 'rec', 'reg'],
            [CLIPLoss, RecLoss, RegLoss]
        ):
            setattr(self, key, c(loss_config.loss_funcs, loss_config.loss_coefs))

    def forward(
        self, batch: tp.Dict[str, tp.Dict]
    ):
        clip_losses = self.clip(batch)
        image_rec_losses = self.rec(batch)
        reg_losses = self.reg(batch)
        
        losses = {**clip_losses, **image_rec_losses, **reg_losses}
        losses['total'] = sum(v for v in losses.values())

        return losses
