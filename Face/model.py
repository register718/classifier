import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class FaceModel(pl.LightningModule):
    
    def __init__(self):
        super(FaceModel, self).__init__()
        ## INPUT 128 x 128
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 64, (7, 7, 3)),
            nn.MaxPool2d((2, 2)),
            nn.Conv3d(64, 64, (5, 5, 5)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (7, 7, 3)),
            nn.MaxPool2d((5, 5)),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3)),
            nn.MaxPool3d((2, 2)),
            nn.Conv3d(64, 32, (3, 3)),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1064, 2000),
            nn.Sigmoid(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000)
        )
    
    def forward(self, x):
        return self.cnn(x)
    
    def tripletLoss(self, est_anchor, est_same, est_other, margin=0.2):
        distance_positive = torch.sum((est_anchor - est_same)**2, dim=1)
        distance_negative = torch.sum((est_anchor - est_other)**2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + margin)
        return torch.mean(loss)
    

    def step(self, batch):
        anchor, same, other = batch
        est_anchor = self(anchor)
        est_same = self(same)
        est_other = self(other)
        loss = self.tripletLoss(est_anchor, est_same, est_other)
        return loss

    def training_step(self, batch):
        return self.step(batch)
    
    def validation_step(self, batch):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)