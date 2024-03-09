import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable

class PositionalEncoder(nn.Module):
  r"""
  Sine-cosine positional encoder for input points.
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Alternate sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    r"""
    Apply positional encoding to input.
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

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

def stratified_sampling(rays, origin, N, t_n, t_f, perturb = True, inverse_depth = False): # also included L
    
    ts = torch.linspace(t_n, t_f, N).to(rays)
    
    if not inverse_depth:
        ts = t_n*(1-ts)  + t_f*ts
    else:
        ts = 1/(1/t_n*(1-ts)  + 1/t_f*ts)
        
    if perturb:
        plane_mid = 0.5*(ts[1:]+ts[:-1])
        plane_near = torch.concat([plane_mid, ts[-1:]], dim=-1)
        plane_far = torch.concat([ts[:1], plane_mid], dim=-1)
        t_rand = torch.rand([N],device = rays.device)
        ts = plane_near + (plane_far - plane_near)*t_rand
        
        
    ts = ts.expand(list(origin.shape[:-1]) + [N])
        

    pts = origin[..., None, :] + rays[..., None, :] * ts[..., :, None]
    
    return pts, ts
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
        
        # print(dists.shape, network_output.shape, ts.shape, rays_d.shape)
        # exit()

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


def Prob_density_func(bins, weights, N, perturb = False):
    
    pdf = (weights+1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)
    
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.concat([torch.zeros_like(cdf[...,:1]), cdf], dim=-1)
    
    
    if not perturb:
        u = torch.linspace(0, 1, N, device= cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N], device= cdf.device)
        
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)   
    
    near = torch.clamp(indices-1, min = 0)
    far = torch.clamp(indices, max = cdf.shape[-1]-1)
    
    indices_g = torch.stack([near, far], dim=-1)
    
    matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=indices_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=indices_g)
    
    
    length = (cdf_g[...,1] - cdf_g[...,0])
    length = torch.where(length < 1e-5, torch.ones_like(length), length)
    
    t = (u - cdf_g[...,0]) / length
    
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    
    return samples

def hierarchical_sampling(origins, rays, ts, weights, N, perturb=False):
    
    plane_mid = 0.5*(ts[...,1:]+ts[...,:-1])
    
    new_ts = Prob_density_func(plane_mid, weights[..., 1:-1], N, perturb=perturb)
    new_ts = new_ts.detach()
    
    ts_combined, _ = torch.sort(torch.cat([ts, new_ts], dim=-1), dim=-1)
    
    points = origins[..., None, :] + rays[..., None, :] * ts_combined[..., :, None]
    
    return points, ts_combined, new_ts
    
    

    
    
    
    
    

    
    
    
    
    
    
    
    
    