import torch
import numpy as np
import tqdm as tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as DataLoader

def positional_encoding(x, L):
    
    gamma_x = [x]
    for i in range(L):
        gamma_x.append(torch.sin((2.0**i)*x))
        gamma_x.append(torch.cos((2.0**i)*x))
    return torch.cat(gamma_x, dim=1)

def RayRendering(model, origin, rays, tn = 2, tf = 6, N=160):
    device = origin.device
    ts = torch.linspace(tn, tf, N, device=device).expand(origin.shape[0], N)
    
    #heirarchal sampling
    plane_mid = (ts[:,:-1]+ts[:,1:])/2
    plane_n = torch.cat((ts[:,:1], plane_mid),-1)
    plane_f = torch.cat((plane_mid, ts[:,-1:]),-1)
    u = torch.rand(ts.shape, device=device)
    t = plane_n + (plane_f - plane_n)*u
    
    
    delta = torch.cat((t[:,1:]-t[:,:-1], torch.tensor([1e10], device=device).expand(origin.shape[0], 1)), dim=-1)
    
    points = origin.unsqueeze(1)+t.unsqueeze(2)*rays.unsqueeze(1)
    rays = rays.expand(N, rays.shape[0], 3).transpose(0,1)
    
    color, sigma = model(points.reshape(-1,3), rays.reshape(-1,3))
    colors = colors.respahe(points.shape)
    sigma = sigma.reshape(points.shape[:-1])
    
    ray_tracing = Volumetric_rendering(sigma, delta).unsqueeze(2) * (1-torch.exp(-sigma*delta)).unsqueeze(2)
    
    c = (ray_tracing*colors).sum(dim=1)
    
    ray_tracing_sum = ray_tracing.sum(dim=-1).sum(-1)
    
    return c + 1-ray_tracing_sum.unsqueeze(-1)
    

def Volumetric_rendering(sigma, delta):
    alpha = 1-torch.exp(-sigma*delta)
    alpha = 1-alpha
    transmittance = torch.cumprod(alpha, dim=1)
    
    return torch.cat((torch.ones((transmittance.shape[0], 1), device=transmittance.device), transmittance[:,:-1]), dim=-1)