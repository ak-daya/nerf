import os

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

from Validation import *
from DataLoader import DataLoader
from Rendering import *
import yaml
from Network import *
from earlystopping import EarlyStopping
from ForwardPass import *



with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f) 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def train():
  r"""
  Initialize models, encoders, and optimizer for NeRF training.
  """
  # Encoders
  encoder = PositionalEncoder(config["Encoders"]["d_input"], config["Encoders"]["n_freqs"], log_space=config["Encoders"]["log_space"])
  encode = lambda x: encoder(x)

  # View direction encoders
  if config["Encoders"]["use_viewdirs"]:
    encoder_viewdirs = PositionalEncoder(config["Encoders"]["d_input"], config["Encoders"]["n_freqs_views"], log_space=config["Encoders"]["log_space"])
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output
  else:
    encode_viewdirs = None
    d_viewdirs = None

  # Models
  model = NeRFModel()
  model.to(device)
  model_params = list(model.parameters())
#   if use_fine_model:
#     fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
#                       d_viewdirs=d_viewdirs)
#     fine_model.to(device)
#     model_params = model_params + list(fine_model.parameters())
#   else:
#     fine_model = None

  # Optimizer
  optimizer = torch.optim.Adam(model_params, lr=config["Optimizer"]["lr"])

  # Early Stopping
  warmup_stopper = EarlyStopping(patience=50)

#   return model, encode, encode_viewdirs, optimizer, warmup_stopper

# def train():
  # Shuffle rays across all images.
  if not config["Training"]["one_image_per_step"]:
    height, width = images.shape[1:3]
    all_rays = torch.stack([torch.stack(generate_rays(height, width, focal, p), 0)
                        for p in poses[:n_training]], 0)
    rays_rgb = torch.cat([all_rays, images[:, None]], 1)
    rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = rays_rgb.reshape([-1, 3, 3])
    rays_rgb = rays_rgb.type(torch.float32)
    rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
    i_batch = 0

  train_psnrs = []
  val_psnrs = []
  iternums = []
  for i in trange(config["Training"]["n_iters"]):
    model.train()

    if config["Training"]["one_image_per_step"]:
      # Randomly pick an image as the target.
      target_img_idx = np.random.randint(images.shape[0])
      target_img = images[target_img_idx].to(device)
      if config["Training"]["center_crop"] and i < config["Training"]["center_crop_iters"]:
        target_img = crop_center(target_img)
      height, width = target_img.shape[:2]
      target_pose = poses[target_img_idx].to(device)
      rays_o, rays_d = generate_rays(height, width, focal, target_pose)
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
    else:
      # Random over all images.
      batch = rays_rgb[i_batch:i_batch + config["Training"]["batch_size"]]
      batch = torch.transpose(batch, 0, 1)
      rays_o, rays_d, target_img = batch
      height, width = target_img.shape[:2]
      i_batch += config["Training"]["batch_size"]
      # Shuffle after one epoch
      if i_batch >= rays_rgb.shape[0]:
          rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
          i_batch = 0
    target_img = target_img.reshape([-1, 3])

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    outputs = nerf_forward(rays_o, rays_d,
                           t_near, t_far, encode, model,
                           kwargs_sample_stratified= None,
                           n_samples_hierarchical=0,
                           kwargs_sample_hierarchical= None,
                           fine_model=None,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=config["Training"]["chunksize"])
    
    # Check for any numerical issues.
    for k, v in outputs.items():
      if torch.isnan(v).any():
        print(f"! [Numerical Alert] {k} contains NaN.")
      if torch.isinf(v).any():
        print(f"! [Numerical Alert] {k} contains Inf.")

    # Backprop!
    rgb_predicted = outputs['rgb_map']
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = -10. * torch.log10(loss)
    train_psnrs.append(psnr.item())

    # Evaluate testimg at given display rate.
    if i % config["Training"]["display_rate"] == 0:
      model.eval()
      height, width = testimg.shape[:2]
      rays_o, rays_d = generate_rays(height, width, focal, testpose)
      rays_o = rays_o.reshape([-1, 3])
      rays_d = rays_d.reshape([-1, 3])
      outputs = nerf_forward(rays_o, rays_d,
                             t_near, t_far, encode, model,
                             kwargs_sample_stratified= None,
                             n_samples_hierarchical=0,
                             kwargs_sample_hierarchical= None,
                             fine_model=model,
                             viewdirs_encoding_fn=encode_viewdirs,
                             chunksize=config["Training"]["chunksize"])

      rgb_predicted = outputs['rgb_map']
      loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
      print("Loss:", loss.item())
      val_psnr = -10. * torch.log10(loss)
      val_psnrs.append(val_psnr.item())
      iternums.append(i)

      # Plot example outputs
      fig, ax = plt.subplots(1, 4, figsize=(24,4), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
      ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
      ax[0].set_title(f'Iteration: {i}')
      ax[1].imshow(testimg.detach().cpu().numpy())
      ax[1].set_title(f'Target')
      ax[2].plot(range(0, i + 1), train_psnrs, 'r')
      ax[2].plot(iternums, val_psnrs, 'b')
      ax[2].set_title('PSNR (train=red, val=blue')
      z_vals_strat = outputs['z_vals_stratified'].view((-1, config["kwargs_sample_stratified"]["n_samples"]))
      z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
      if 'z_vals_hierarchical' in outputs:
        z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, config["Hierarchical_sampling"]["n_samples_hierarchical"]))
        z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
      else:
        z_sample_hierarch = None
      _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
      ax[3].margins(0)
      plt.show()

    # Check PSNR for issues and stop if any are found.
    if i == config["Early_Stopping"]["warmup_iters"] - 1:
      if val_psnr < config["Early_Stopping"]["warmup_min_fitness"]:
        print(f'Val PSNR {val_psnr} below warmup_min_fitness {config["Early_Stopping"]["warmup_min_fitness"]}. Stopping...')
        return False, train_psnrs, val_psnrs
    elif i < config["Early_Stopping"]["warmup_iters"]:
      if warmup_stopper is not None and warmup_stopper(i, psnr):
        print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
        return False, train_psnrs, val_psnrs
    
  return model, encode, encode_viewdirs, optimizer, warmup_stopper, True, train_psnrs, val_psnrs

for _ in range(config["Early_Stopping"]["n_restarts"]):
  success, train_psnrs, val_psnrs, model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = train()
  if success and val_psnrs[-1] >= config["Early_Stopping"]["warmup_min_fitness"]:
    print('Training successful!')
    break

print('')
print(f'Done!')

torch.save(model.state_dict(), 'nerf.pt')
torch.save(fine_model.state_dict(), 'nerf-fine.pt')