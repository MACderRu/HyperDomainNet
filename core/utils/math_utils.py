import torch


def resample_single_vector(target_vector, cos_lower_bound, n_vectors=1):
    """
    Resample one vector 'n_vectors' times with lower bound of cos 'cos_lower_bound'

    Parameters
    ----------
    target_vector : torch.Tensor with size() == (1, dim) || (dim)
        center of resampling
    cos_lower_bound : float
        lower bound of cos of resampled vectors
    n_vectors : int
        number of resampled vectors

    Returns
    -------
    omega : torch.Tensor, size [n_vectors, vector_dim]
        resampled vectors with cos with target_vector higher than thr_cos
    """

    if target_vector.ndim == 1:
        target_vector = target_vector.unsqueeze(0)

    _, dim = target_vector.size()

    u = target_vector / target_vector.norm(dim=-1, keepdim=True)
    u = u.repeat(n_vectors, 1)
    r = torch.rand_like(u) * 2 - 1
    uperp = torch.stack([r[i] - (torch.dot(r[i], u[i]) * u[i]) for i in range(u.size(0))])
    uperp = uperp / uperp.norm(dim=1, keepdim=True)

    cos_theta = torch.rand(n_vectors, device=target_vector.device) * (1 - cos_lower_bound) + cos_lower_bound
    cos_theta = cos_theta.unsqueeze(1).repeat(1, target_vector.size(1))
    omega = cos_theta * u + torch.sqrt(1 - cos_theta ** 2) * uperp

    return omega


def resample_batch_vectors(target_vector, cos_lower_bound):
    """
    Resample 'b' vector 'n_vectors' times with lower bound of cos 'cos_lower_bound'

    Parameters
    ----------
    target_vector : torch.Tensor with size() == (b, dim)
        center of resampling
    cos_lower_bound : float
        lower bound of cos of resampled vectors

    Returns
    -------
    omega : torch.Tensor, size [n_vectors, vector_dim]
        resampled vectors with cos with target_vector higher than thr_cos
    """

    b, dim = target_vector.size()
    u = target_vector / target_vector.norm(dim=-1, keepdim=True)
    r = torch.rand_like(u) * 2 - 1
    uperp = torch.stack([r[i] - (torch.dot(r[i], u[i]) * u[i]) for i in range(u.size(0))])
    uperp = uperp / uperp.norm(dim=1, keepdim=True)

    cos_theta = torch.rand(b, device=target_vector.device) * (1 - cos_lower_bound) + cos_lower_bound
    cos_theta = cos_theta.unsqueeze(1).repeat(1, target_vector.size(1))
    omega = cos_theta * u + torch.sqrt(1 - cos_theta ** 2) * uperp

    return omega


def resample_batch_templated_embeddings(embeddings, cos_lower_bound):
    if embeddings.ndim == 2:
        return resample_batch_vectors(embeddings, cos_lower_bound)

    batch, templates, dim = embeddings.shape
    embeddings = embeddings.view(-1, dim)
    resampled_embeddings = resample_batch_vectors(embeddings, cos_lower_bound)

    resampled_embeddings = resampled_embeddings.view(batch, templates, dim).contiguous()
    return resampled_embeddings


def convex_hull(target_vectors, alphas):
    """
    calculate convex hull with 'alphas' (1 > alpha > 0, \sum alphas = 1) for target vectors

    Parameters
    ----------
    target_vectors : torch.Tensor
        set of vectors for which convex hull element is calculated.
        Size: [b, dim1, dim2]

    alphas : torch.Tensor
        appropriate alphas for which element from convex hull will be calculated.
        Size: [b, b]

    Returns
    -------
    convex_hull_element : torch.Tensor
        single element from convex hull

    """

    convex_hull_element = (target_vectors.unsqueeze(0) * alphas.unsqueeze(2).unsqueeze(3)).sum(dim=1)
    convex_hull_element /= convex_hull_element.clone().norm(dim=-1, keepdim=True)
    return convex_hull_element


def convex_hull_small(target_vectors, alphas):
    """
    calculate convex hull with 'alphas' (1 > alpha > 0, \sum alphas = 1) for target vectors

    Parameters
    ----------
    target_vectors : torch.Tensor
        set of vectors for which convex hull element is calculated.
        Size: [b, dim1, dim2]

    alphas : torch.Tensor
        appropriate alphas for which element from convex hull will be calculated.
        Size: [b, b]

    Returns
    -------
    convex_hull_element : torch.Tensor
        single element from convex hull

    """

    convex_hull_element = (target_vectors.unsqueeze(0) * alphas.unsqueeze(2)).sum(dim=1)
    convex_hull_element /= convex_hull_element.clone().norm(dim=-1, keepdim=True)
    return convex_hull_element
