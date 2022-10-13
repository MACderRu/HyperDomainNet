import math

import PIL
import scipy.ndimage
import clip
import dlib
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms




def requires_grad(model, flag=True):
    if isinstance(model, nn.Parameter):
        model.requires_grad = flag
        return

    for p in model.parameters():
        p.requires_grad = flag


def load_clip(model_name: str, device: str):
    """
    Get clip model with preprocess which can process torch images in value range of [-1, 1]

    Parameters
    ----------
    model_name : str
        CLIP-encoder type

    device : str
        Device for clip-encoder

    Returns
    -------
    model : nn.Module
        torch model of downloaded clip-encoder

    preprocess : torchvision.transforms.transforms
        image preprocess for images from stylegan2 space to clip input image space
            - value range of [-1, 1] -> clip normalized space
            - resize to 224x224
    """
    model, preprocess = clip.load(model_name, device=device)
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
        *preprocess.transforms[:2],
        preprocess.transforms[-1]
    ])

    return model, preprocess


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


def validate_device(device_to_validate):
    if device_to_validate == "cpu":
        device = "cpu"
    elif device_to_validate.isdigit():
        device = "cuda:{}".format(device_to_validate)
    elif device_to_validate.startswith("cuda:"):
        device = device_to_validate
    else:
        raise ValueError("Incorrect Device Type")

    try:
        torch.randn(1, device=device)
        print("Device: {}".format(device))
    except Exception as e:
        print("Could not use device {}, {}".format(device, e))
        print("Device is set to CPU")
        device = "cpu"

    return device


def get_valid_exp_dir_name(base_root, exp_name, exp_root="local_logged_exps"):
    base_experiments_root = base_root / exp_root

    num = 0
    exp_dir = "{}_{}".format(exp_name, str(num).zfill(3))

    exp_path = base_experiments_root / exp_dir
    while exp_path.exists():
        num += 1
        exp_dir = "{}_{}".format(exp_name, str(num).zfill(3))
        exp_path = base_experiments_root / exp_dir

    return exp_path


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def compose_text_with_templates(text, templates=("A {}", )):
    return [s.format(text) for s in templates]


def read_domain_list(path):
    with open(path, 'r') as f:
        return [line.strip().split(' - ') for line in f]


def read_image_list(path):
    image_list = []
    with open(path, 'r') as f:
        for image_path in f:
            image_path = image_path.strip()
            name = image_path.split("/")[-1].split(".")[0]
            image_list.append((name, image_path))
    return image_list


def read_style_images_list(path):
    with open(path, 'r') as f:
        return [p.strip() for p in f]


def determine_opt_layers(
    source_generator,
    trainable,
    clip_loss,
    config,
    target_class,
    auto_layer_iters,
    auto_layer_batch,
    auto_layer_k,
    device='cuda'
):
    sample_z = torch.randn(auto_layer_batch, 512, device=device)
    initial_w_codes = source_generator.style([sample_z])
    initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, source_generator.generator.n_latent, 1)

    w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(device)
    w_codes.requires_grad = True
    w_optim = torch.optim.SGD([w_codes], lr=0.01)

    for _ in range(auto_layer_iters):
        w_codes_for_gen = w_codes.unsqueeze(0)

        if config.training.patch_key == "original":
            generated_from_w, _ = trainable(w_codes_for_gen, input_is_latent=True)
        else:
            generated_from_w, _ = source_generator(
                w_codes_for_gen,
                weights_deltas=trainable(),
                input_is_latent=True
            )

        w_loss = clip_loss.global_clip_loss(generated_from_w, clip_loss.target_embeddings[target_class])

        w_optim.zero_grad()
        w_loss.backward()
        w_optim.step()

    layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)

    all_layers = list(trainable.get_all_layers())

    if config.training.patch_key == 'original':
        chosen_layer_idx = torch.topk(layer_weights, auto_layer_k)[1].cpu().numpy()
        idx_to_layer = all_layers[2:4] + list(all_layers[4])
        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx]
    else:
        chosen_layer_idx = torch.topk(layer_weights[:-1], auto_layer_k)[1].cpu().numpy()
        chosen_layers = [all_layers[idx] for idx in chosen_layer_idx]

    return chosen_layers


def get_stylegan_conv_dimensions(size, channel_multiplier=2):
    channels = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256 * channel_multiplier,
        128: 128 * channel_multiplier,
        256: 64 * channel_multiplier,
        512: 32 * channel_multiplier,
        1024: 16 * channel_multiplier,
    }

    log_size = int(math.log(size, 2))
    conv_dimensions = [(channels[4], channels[4])]

    in_channel = channels[4]

    for i in range(3, log_size + 1):
        out_channel = channels[2 ** i]

        conv_dimensions.append((in_channel, out_channel))
        conv_dimensions.append((out_channel, out_channel))

        in_channel = out_channel

    return conv_dimensions


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    assert len(dets) > 0, "Face not detected, try another face image"

    for k, d in enumerate(dets):
        shape = predictor(img, d)
    lm = np.array([[tt.x, tt.y] for tt in shape.parts()])
    return lm


def align_face(filepath, predictor, output_size=1024, transform_size=4096, enable_padding=True):
    """
    :param filepath: str
    :return: list of PIL Images
    """

    lm = get_landmark(filepath, predictor)
    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_trainable_model_state(config, state_dict):
    if config.training.patch_key == "original":
        # Save TunningGenerator as state_dict
        ckpt = {
            "model_type": 'original',
            "state_dict": state_dict
        }
    elif config.get("mapper_config", False):
        # save mapper
        ckpt = {
            "model_type": "mapper",
            "mapper_config": config.training.mapper_config,
            "patch_key": config.training.patch_key,
            "state_dict": state_dict,
        }
    else:
        # save offsets parametrization
        ckpt = {
            "model_type": "parameterization",
            "patch_key": config.training.patch_key,
            "state_dict": state_dict
        }
    
    ckpt['sg_2_params'] = dict(config.generator_args['stylegan2'])
    return ckpt


# def build_from_checkpoint(ckpt, generator_size=1024, generator_latent_dim=512, generator_nmlp=8):
#     assert ckpt['model_type'] in ['original', 'mapper', 'parametrization']

#     if ckpt['model_type'] == "original":
#         model = OffsetsTunningGenerator(
#             img_size=generator_size,
#             latent_size=generator_latent_dim,
#             map_layers=generator_nmlp
#         )
#         model.generator.load_state_dict(ckpt['state_dict'])
#     elif ckpt['model_type'] == "mapper":
#         model = mapper_registry[ckpt['mapper_type']](
#             ckpt['mapper_config'],
#             get_stylegan_conv_dimensions(generator_size)
#         )

#         model.load_state_dict(ckpt['state_dict'])
#     else:
#         model = BaseParametrization(
#             ckpt['base_head_key'],
#             get_stylegan_conv_dimensions(generator_size)
#         )
#         model.load_state_dict(ckpt['state_dict'])

#     return model
