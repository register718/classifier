import trainer as T
import model.net1 as nets
import torch
import datasets.MINSTDataLoader as minstSET
import matplotlib.pyplot as plt

torch.manual_seed(0)

net = nets.NetOne()
EPOCHS = 100
### Data Sets ##

trainSet = minstSET.MNISTDataLoaderMix(train=True)
testSet = minstSET.MINSTDataLoaderMax(train=False)

tMinMax = T.MinMaxTrainer(net, trainSet, testSet, EPOCHS=EPOCHS)
tMin = T.MinTrainer(net, trainSet, testSet, EPOCHS=EPOCHS)
tMax = T.MaxTrainer(net, trainSet, testSet, EPOCHS=EPOCHS)

def showPoints(t):
    if isinstance(t, list):
        n = torch.cat(t, dim=1).detach()
    else:
        n = t.detach()
    n = n.numpy()
    n1 = n[:,0]; n2 = n[:,1]
    plt.plot(n1, n2, 'ro')
    plt.show(block=True)

def trainTwoPoints(p1, p2, trainer: T.Trainer):
    x = p1, 0, p2, 0
    trainer.trainingStep(x)

p = trainSet[0]

in1, l1, in2, l2 = p
in1 = in1.unsqueeze(dim=0)
in2 = in2.unsqueeze(dim=0)

while True:
    showPoints([net(in1), net(in2)])
    for i in range(100):
        trainTwoPoints(in1, in2, tMax)
    print(net(in1), net(in2))
