import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class SiameseDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img1, label1 = self.mnist_dataset[idx]
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                idx2 = random.randint(0, len(self.mnist_dataset) - 1)
                img2, label2 = self.mnist_dataset[idx2]
                if label1 == label2:
                    break
        else:
            while True:
                idx2 = random.randint(0, len(self.mnist_dataset) - 1)
                img2, label2 = self.mnist_dataset[idx2]
                if label1 != label2:
                    break

        return (img1, img2), torch.tensor(int(label1 == label2), dtype=torch.float32)

class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward_one(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2

    def contrastive_loss(self, embedding1, embedding2, label, margin=2.0):
        euclidean_distance = nn.functional.pairwise_distance(embedding1, embedding2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + 
                                       (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    def training_step(self, batch, batch_idx):
        (img1, img2), label = batch
        embedding1, embedding2 = self(img1, img2)
        loss = self.contrastive_loss(embedding1, embedding2, label)
        #self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx)
            self.log('val_loss', loss)
            return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)

def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset_train = torchvision.datasets.MNIST(root='../MaxMin/data', train=True, transform=transform, download=True)
    minst_dataset_test = torchvision.datasets.MNIST(root='../MaxMin/data', train=False, transform=transform, download=True)
    siamese_dataset_train = SiameseDataset(mnist_dataset_train)
    siamese_dataset_test = SiameseDataset(minst_dataset_test)
    train_loader = DataLoader(siamese_dataset_train, batch_size=64, shuffle=True, num_workers=7)
    test_loader = DataLoader(siamese_dataset_test, batch_size=16, num_workers=7, shuffle=False)

    # Create an instance of the SiameseNetwork model
    model = SiameseNetwork()

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min")
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=acc, devices=1, max_epochs=20,
                            accumulate_grad_batches=64,
                            callbacks=[early_stop_callback], detect_anomaly=False)

    # Train the model
    trainer.fit(model, train_loader, val_dataloaders=test_loader)

    torch.save(model.state_dict(), "./weights/test.pt")

if __name__=='__main__':main()