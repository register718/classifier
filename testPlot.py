import torch
import model.net1 as nets
import datasets.MINSTDataLoader as minstSET
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

torch.manual_seed(0)

torch.set_grad_enabled(False)

net = nets.NetOne()

NUM_TARGETS = 10

net.load_state_dict(torch.load("./weights/plot_mnist_20.pt"))

test_set = minstSET.MNISTDataLoaderMin(train=False)
test_loader = DataLoader(test_set, shuffle=True)

results = []
_labels = []

for x in test_loader:
    y1 = net(x[0]); y2 = net(x[2])
    _labels.append(x[1]); _labels.append(x[3])
    results.append(y1); results.append(y2)

output = np.array(results)[:,0,:]
labels = np.array(_labels)[:,0]

kmeans = KMeans(n_clusters=NUM_TARGETS, random_state=0, n_init="auto").fit(output)

res = np.count_nonzero(kmeans.labels_ == labels)
print(res, res / len(labels))
print(labels[kmeans.labels_ == labels])
print(kmeans.labels_[kmeans.labels_==labels])