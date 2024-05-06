import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import Inception3, Inception_V3_Weights

class FaceModel(pl.LightningModule):
    
    def __init__(self):
        ## INPUT 64 x 64
        pretrained = Inception3(weights=Inception_V3_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            pretrained,
            nn.Flatten()
        )