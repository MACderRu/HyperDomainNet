import torch
import torch.nn.functional as F
import numpy as np


def cosine_loss(x, y):
    return 1.0 - F.cosine_similarity(x, y)


def mse_loss(x, y):
    return F.mse_loss(x, y)


def mae_loss(x, y):
    return F.l1_loss(x, y)


def get_tril_elemets(matrix_torch: torch.Tensor):
    flat = torch.tril(matrix_torch, diagonal=-1).flatten()
    return flat[torch.nonzero(flat)]


def get_tril_elements_mask(linear_size):
    mask = np.zeros((linear_size, linear_size), dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    np.fill_diagonal(mask, False)
    return mask


def flatten_with_non_diagonal(input_matix: torch.Tensor):
    linear_matrix_size = input_matix.size(0)

    non_diag = input_matix.flatten()[1:].view(linear_matrix_size - 1, linear_matrix_size + 1)[:, :-1]
    return non_diag.flatten()
