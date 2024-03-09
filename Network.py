import torch.nn as nn
import torch
from torchsummary import summary

class NeRFModel(nn.Module):
    def __init__(self,
        layers = 8,
        ch_hidden = 256,
        ch_pos = 3,
        ch_dir = 3,
        Freq_pos = 10,
        Freq_dir = 4
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(6*Freq_pos+3, ch_hidden),
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
            nn.Linear(ch_hidden + 6*Freq_pos+3, ch_hidden),
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
            nn.Linear(ch_hidden + 6*Freq_dir+3, ch_hidden // 2),
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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = NeRFModel().to(device)

dummyinput_x, dummyinput_d = (torch.rand(1, 63).to(device), torch.rand(1, 27).to(device))  

out = model(dummyinput_x, dummyinput_d) 

print(out.shape)
print(out)

# out.view(800, 800, 8, 4)

summary(model, [(63,), (27,)])
