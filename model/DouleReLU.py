import torch

class DoubleReLU(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x = torch.where(torch.abs(x) >= 1, torch.sgn(x), x)
        return x