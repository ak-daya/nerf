import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable

def positional_encoding(x, L):
    gamma_x = [x]
    for i in range(L):
        gamma_x.append(torch.sin(2**i * x)) # removed pi term
        gamma_x.append(torch.cos(2**i * x)) # removed pi term
        
    return torch.cat(gamma_x, axis=-1)

# def raygen(H, W, focal_length, T):
#     """
#     Generates rays for rendering.

#     Args:
#         H (int): Height of the image.
#         W (int): Width of the image.
#         focal_length (float): Focal length of the camera.
#         T (torch.Tensor): Transformation matrix of the camera.
#         N (int): Number of rays to generate.
#         t_n (float, optional): Near clipping distance. Defaults to 2.
#         t_f (float, optional): Far clipping distance. Defaults to 6.
#         N_rand_rays (int, optional): Number of random rays to generate. Defaults to None.
#         n_frequencies (int, optional): Number of frequencies for stratified sampling. Defaults to 4.
#         rand (bool, optional): Whether to generate random rays. Defaults to False.

#     Returns:
#         points and samples
#     """
#     # Function implementation goes here
    
#     rotation_matrix = T[:3, :3]
#     translation_vector = T[:3, 3]
    
#     xs = torch.linspace(0, H-1, H).to(device)
#     ys = torch.linspace(0, W-1, W).to(device)
    
#     x, y = torch.meshgrid(xs, ys)
    
#     x, y = x.t(), y.t()
    
#     X = (x-H/2)/focal_length
#     Y = (y-W/2)/focal_length
    
#     rays = torch.stack([X, -Y, -torch.ones_like(X)], axis=-1)
    
#     rays = rays.view(H, W, 1, 3)
    
#     # transform to camera pose
    
#     rays = torch.matmul(rays, rotation_matrix.t())
    
#     # Normalizing the vectors to be unit vectors
#     rays = rays / torch.norm(rays, dim=-1, keepdim=True)
    
#     rays = rays.view(H, W, 3)
#     origin = torch.broadcast_to(translation_vector, rays.shape)
    
#     return rays, origin

# Another ray gen implementation
def generate_rays(
    height: int, 
    width: int,
    focal_length: float, 
    transform: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find origin and direction of rays through every pixel and camera origin.
    """

    # Apply pinhole camera model to gather directions at each pixel
    x = torch.arange(width, dtype=torch.float32).to(transform)
    y = torch.arange(height, dtype=torch.float32).to(transform)

    i, j = torch.meshgrid(x, y, indexing='ij')
    # Flip order of i,j 
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    
    # Normalized image coordinates
    directions = torch.stack([(i - width * .5) / focal_length,
                                -(j - height * .5) / focal_length,
                                -torch.ones_like(i)
                            ], dim=-1)

    # Transform pixel directions according to cam pose 
    # Uses hadamard product to individually apply rotation
    rotation = transform[:3, :3]
    rays_d = torch.sum(directions[:, :, None, :] * rotation, dim=-1)

    # Ray origin is the optical center
    rays_o = transform[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d

def cumprod_exclusive(tensor):
    """
    Mimic tf.math.cumprod(..., exclusive=True)
    """
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.

    return cumprod

def stratified_sampling(rays, origin, N, t_n, t_f): # also included L
    
    ts = torch.linspace(t_n, t_f, N).to(rays)
    ts = ts.expand(list(origin.shape[:-1]) + [N])

    pts = origin[..., None, :] + rays[..., None, :] * ts[..., :, None]
    # pts = pts.view(-1, 3)

    
    # encoded_pts = PositionalEncoding(pts, L)
    
    return pts, ts #encoded_pts, ts

# WIP:
def PrepareForVolumeRendering(
    network_output: torch.Tensor,
    ts: torch.Tensor,
    rays_d: torch.Tensor,
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the raw NeRF output into RGB and other maps.
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = ts[..., 1:] - ts[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if noise_std > 0.:
            noise = torch.randn(network_output[..., 3].shape) * noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        alpha = 1.0 - torch.exp(-nn.functional.relu(network_output[..., 3] + noise) * dists)

        # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
        # The higher the alpha, the lower subsequent weights are driven.
        weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

        # Compute weighted RGB map.
        rgb = torch.sigmoid(network_output[..., :3])  # [n_rays, n_samples, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

        # Estimated depth map is predicted distance.
        depth_map = torch.sum(weights * ts, dim=-1)

        # Sum of weights along each ray. In [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        return rgb_map, depth_map, acc_map, weights

# rays, origin = raygen(10, 10, 1, torch.eye(4))

# encoded_pts, ts = pointsgen(rays, origin, 10, 2, 6, 10)

    
    
    
    
    

    
    
    
    
    
    
    
    
    