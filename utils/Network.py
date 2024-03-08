import torch
import numpy as np
import tqdm as tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as DataLoader
from RayGen import *

class MLP(nn.Module):
    def __init__(self, encoded_points_dims=10, encoded_d_dims = 4):
        super(MLP, self).__init__()
        
        self.encoded_points_dims = encoded_points_dims
        self.encoded_d_dims = encoded_d_dims
        
        self.net1 = nn.Sequential(nn.Linear(encoded_points_dims*6+3, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),)
        
        self.net2 = nn.Sequential(nn.Linear(encoded_points_dims*6+256+3, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 257), nn.ReLU(),)
        
        self.net3 = nn.Sequential(nn.Linear(encoded_d_dims*6+256+3, 128), nn.ReLU(),)
        self.net4 = nn.Sequential(nn.Linear(128, 3), nn.ReLU(),)
        
    def forward(self, points, d):
        
        encoded_points = positional_encoding(points, self.encoded_points_dims)
        encoded_d = positional_encoding(d, self.encoded_d_dims)
        temp1 = self.net1(encoded_points)
        temp2 = self.net2(torch.cat((encoded_points, temp1), dim=1))
        temp3, sigma = temp2[:, :-1], nn.ReLU(temp2[:, -1])
        temp4 = self.net3(torch.cat((encoded_d, temp3), dim=1))
        rgb = self.net4(temp4)
        
        return rgb, sigma
        
        
        
    
        
        
                                     