import trainer as T
import model.net1 as nets
import torch
import datasets.MINSTDataLoader as minstSET


torch.manual_seed(0)

net = nets.NetOne()
### Data Sets ##

trainSet = minstSET.MNISTDataLoaderMix(train=True)
testSet = minstSET.MINSTDataLoaderMax(train=False)

EPOCHS = 1000
trainer = T.MinMaxTrainer(net, trainSet, testSet, EPOCHS=EPOCHS)

##########
import plot

plot = plot.Plot(trainer, 10)
plot.start_without_animation()
torch.save(net.state_dict(), f"./weights/plot_mnist_{str(EPOCHS)}.pt")