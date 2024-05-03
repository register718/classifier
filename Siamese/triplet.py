import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Siamese Network Definition
class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward_one(self, x):
        return self.cnn(x)

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

    def triplet_loss(self, anchor, positive, negative, margin=0.1):
        distance_positive = torch.sum((anchor - positive)**2, dim=1)
        distance_negative = torch.sum((anchor - negative)**2, dim=1)
        loss = torch.relu(distance_positive - distance_negative + margin)
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        anchor_img, positive_img, negative_img = batch
        anchor_out, positive_out, negative_out = self(anchor_img, positive_img, negative_img)
        loss = self.triplet_loss(anchor_out, positive_out, negative_out)
        #self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            anchor_img, positive_img, negative_img = batch
            anchor_out, positive_out, negative_out = self(anchor_img, positive_img, negative_img)
            loss = self.triplet_loss(anchor_out, positive_out, negative_out)
            self.log('val_loss', loss)
            return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Custom Dataset for Triplets
class TripletMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        anchor_img, anchor_label = self.mnist_dataset[idx]
        
        # Find positive sample (same class)
        positive_idx = np.random.choice(np.where(self.mnist_dataset.targets.numpy() == anchor_label)[0])
        positive_img, _ = self.mnist_dataset[positive_idx]

        # Find negative sample (different class)
        negative_idx = np.random.choice(np.where(self.mnist_dataset.targets.numpy() != anchor_label)[0])
        negative_img, _ = self.mnist_dataset[negative_idx]

        return anchor_img, positive_img, negative_img

def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = torchvision.datasets.MNIST(root='../MaxMin/data', train=True, transform=transform, download=True)
    minst_dataset_test = torchvision.datasets.MNIST(root='../MaxMin/data', train=False, transform=transform, download=True)
    tripletDataSetTest = TripletMNISTDataset(minst_dataset_test)
    test_loader = DataLoader(tripletDataSetTest, batch_size=16, num_workers=7, shuffle=False)

    triplet_mnist_dataset = TripletMNISTDataset(mnist_dataset)
    train_loader = DataLoader(triplet_mnist_dataset, batch_size=64, shuffle=True, num_workers=7)

    # Create a PyTorch Lightning Trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=acc, devices=1, max_epochs=50,
                            accumulate_grad_batches=64,
                            callbacks=[early_stop_callback], detect_anomaly=False)
    
    model = SiameseNetwork()

    # Train the model
    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    torch.save(model.state_dict(), "./weights/triplet0.1.pt")

if __name__=='__main__':main()
