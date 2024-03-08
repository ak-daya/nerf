import torch.nn as nn
import torch

class NeRFModel(nn.Module):
    def __init__(self,
        layers = 8,
        ch_hidden = 256,
        ch_pos = 3,
        ch_dir = None,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(ch_pos, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            # Skip connection
            nn.Linear(ch_hidden + ch_dir, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden),
            nn.ReLU(),
            nn.Linear(ch_hidden, ch_hidden)
        )

        self.opacity_out = nn.Sequential(
            nn.Linear(ch_hidden, 1), 
            nn.ReLU())
        
        self.block3 = nn.Sequential(
            nn.Linear(ch_hidden + ch_dir, ch_hidden // 2),
            nn.ReLU()
        )

        self.color_out = nn.Sequential(
            nn.Linear(ch_hidden // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, d):
        """
        Forward pass of network
        Feed position, x, direction, d
        """
        # Residual
        res = x

        x = self.block1(x)
        
        # Skip connection
        x = torch.cat([x, res], dim=-1)
        x = self.block2(x)
        
        # Opacity
        opacity = self.opacity_out(x)

        # Color
        x = torch.cat([x, d], dim=-1)
        x = self.block3(x)
        color = self.color_out(x)

        # 4-d vector containing 3-d color and 1-d opacity
        x = torch.cat([color, opacity], dim=-1)

        return x
