import trainer as T
import model.net1 as nets
import torch
import datasets.MINSTDataLoader as minstSET
import matplotlib.pyplot as plt

torch.manual_seed(0)

net = nets.NetOne()
EPOCHS = 100
### Data Sets ##

trainMix = minstSET.MNISTDataLoaderMix(train=False)
trainMin = minstSET.MNISTDataLoaderMin(train=False)
trainMax = minstSET.MINSTDataLoaderMax(train=False)

tMinMax = T.MinMaxTrainer(net, trainMix, trainMix, EPOCHS=EPOCHS)
tMin = T.MinTrainer(net, trainMin, trainMin, EPOCHS=EPOCHS)
tMax = T.MaxTrainer(net, trainMax, trainMax, EPOCHS=EPOCHS)

def showPoints(t):
    if isinstance(t, list):
        n = torch.cat(t, dim=0).detach()
    else:
        n = t.detach()
    n = n.numpy()
    n1 = n[:,0]; n2 = n[:,1]
    plt.plot(n1, n2, 'ro')
    plt.show(block=True)

def trainTwoPoints(p1, p2, trainer: T.Trainer):
    x = p1, 0, p2, 0
    return trainer.trainingStep(x)


for i in range(10):
    p = trainMax[i][0].unsqueeze(dim=0)
    u = trainMin[i][0].unsqueeze(dim=0)
    showPoints([net(p), net(u)])
    N = 100
    sum = 0
    for i in range(100):
        sum += trainTwoPoints(p, u, tMax)
    print(net(p), net(u), sum / N)
