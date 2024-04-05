import torch
import torch.nn.functional as F
from .DouleReLU import DoubleReLU

class NetOne(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)
        self.DReLU = DoubleReLU()
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # [6, 24,24] -> [6, 12, 12]
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) # [16, 10, 10] nach max pool -> [16, 5, 5]
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.DReLU(x)
        return x

        