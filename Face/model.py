import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import Inception3, Inception_V3_Weights

class FaceModel(pl.LightningModule):
    
    def __init__(self):
        ## INPUT 128 x 128
        pretrained = Inception3(weights=Inception_V3_Weights.DEFAULT)
        self.cnn = nn.Sequential(
            pretrained,
            nn.Flatten(),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.cnn(x)
    
    def tripletLoss(self, est_anchor, est_same, est_othern, margin=0.2):
        distance_positive = torch.sum((est_anchor - est_same)**2, dim=1)
        distance_negative = torch.sum((est_anchor - est_other)**2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + margin)
        return torch.mean(loss)
    

    def step(self, batch):
        anchor, same, other = batch
        est_anchor = self(anchor)
        est_same = self(same)
        est_other = self(other)
        loss = self.tripletLoss(est_anchor, est_same, est_othern)
        return loss

    def training_step(self, batch):
        return self.step(batch)
    
    def validation_step(self, batch):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)