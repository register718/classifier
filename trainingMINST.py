import torch
import datasets.MINSTDataLoader as minstSET
from torch import optim
from torch.utils.data import DataLoader

torch.manual_seed(0)

trainSet = minstSET.MNISTDataLoaderMix(train=True)
train_loader = DataLoader(trainSet, shuffle=True, batch_size=8)
testSet = minstSET.MNISTDataLoaderMix(train=False)
test_loader = DataLoader(testSet, shuffle=False)

import model.net1 as net
net = net.NetOne()

def train_loop(train_data, test_data):
    learning_rate = 1e-3
    EPOCHS = 10

    optimizerMin = optim.Adam(net.parameters(), maximize=False, lr=learning_rate)
    optimizerMax = optim.Adam(net.parameters(), maximize=True, lr=learning_rate)

    # TRAIN EPOCH
    for u in range(0, EPOCHS):
        print("Starte Epoche", u)
        for x in train_data:
            in1, l1, in2, l2 = x
            y1 = net(in1); y2 = net(in2)
            loss = torch.sum((y1-y2).pow(2))
            loss.backward()
            optimizerMin.step()
            optimizerMin.zero_grad()
        with torch.no_grad():
            losses = []
            for x in test_data:
                in1, l1, in2, l2 = x
                y1 = net(in1); y2 = net(in2)
                loss = torch.sum((y1-y2).pow(2))
                losses.append(loss)
            l = torch.tensor(losses)
            print("Loss", l.mean())

train_loop(train_loader, test_loader)
