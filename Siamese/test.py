import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import network

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = torchvision.datasets.MNIST(root='../MaxMin/data', train=False, transform=transform, download=True)
siamese_dataset = network.SiameseDataset(mnist_dataset)
test_loader = DataLoader(siamese_dataset, batch_size=1, shuffle=False)

# Load trained Siamese Network
model = network.SiameseNetwork()  # Replace 'path_to_saved_model.ckpt' with the actual path
weights = torch.load("./weights/test.pt")
model.load_state_dict(weights)

# Send test data through the network
labels = []
similar=[]
dissimilar = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Processing batches'):
        label = batch[1]; img1, img2 = batch[0]
        est_img1 = model.forward_one(img1)
        est_img2 = model.forward_one(img2)
        dist = torch.nn.functional.pairwise_distance(est_img1, est_img2)
        if label == 0:
            similar.append(dist)
        else:
            dissimilar.append(dist)

# Concatenate embeddings from all batches into a single tensor
import numpy as np
import matplotlib.pyplot as plt

similar = torch.cat(similar, dim=0).numpy()
dissimilar = torch.cat(dissimilar, dim=0).numpy()
counts, bins = np.histogram(similar)
plt.stairs(counts, bins, color="green")
counts2, bins2 = np.histogram(dissimilar)
plt.stairs(counts2, bins2, color="blue")
plt.show()


