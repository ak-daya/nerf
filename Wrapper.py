import os

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

from Validation import *
from DataLoader import DataLoader
from utils.Rendering import stratified_sampling, positional_encoding, cumprod_exclusive, generate_rays

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import data
    data = DataLoader("lego")

    # Show sample outputs from DataLoader
    # show_sample_outputs_from_dataloader(data)

    n_training = data.train.images.shape[0]
    testimg, testpose = data.test.images[0], data.test.transforms[0]

    # Visualize camera poses in 3D
    # visualize_poses_in_3d(data)
    
    # Data to tensors
    images = torch.from_numpy(data.train.images).to(device)
    poses = torch.from_numpy(data.train.transforms).to(device)
    focal = torch.as_tensor(138.8889).to(device)
    testimg = torch.from_numpy(testimg).to(device)
    testpose = torch.from_numpy(testpose).to(device)

    N_stratified_samples = 8
    t_near, t_far = 2., 6.
    h, w = data.train.images.shape[1:3]
    
    with torch.no_grad():
        # test only
        ray_origin, ray_direction = generate_rays(h, w, focal, testpose)
        
        rays_o = ray_origin.view([-1,3])
        rays_d = ray_direction.view([-1,3])

        pts, ts = stratified_sampling(rays_d, rays_o, N_stratified_samples, t_near, t_far)
    
    # Visualize stratified sampling
    # plot_stratified_sampling(ts)
        
    rays_d_normalized = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    x_flattened = pts.reshape((-1,3))
    d_flattened = rays_d_normalized[:, None, ...].expand(pts.shape).reshape(-1,3)

    # Positional encoding - send input to higher dim
    x_encoded = positional_encoding(x_flattened, 10)
    d_encoded = positional_encoding(d_flattened, 4)

    # Show shape and sample outputs from encoding
    # show_sample_outputs_from_positional_encoding(x_encoded, d_encoded)
    
    # To Do: Volume Rendering, Nerf forward pipeline, train