import trainer as T
import model.net1 as nets
import torch
import datasets.MINSTDataLoader as minstSET


torch.manual_seed(0)

net = nets.NetOne()
### Data Sets ##

EPOCHS = 300
trainer = T.MinMaxTrainer(net, minstSET.MNISTDataLoaderMix(train=True), minstSET.MINSTDataLoaderMax(train=False), EPOCHS=EPOCHS)
#trainer = T.MaxTrainer(net, minstSET.MINSTDataLoaderMax(train=True), minstSET.MINSTDataLoaderMax(train=False), EPOCHS=EPOCHS)

##########
import plot

plot = plot.Plot(trainer, 10)
plot.start_without_animation()
torch.save(net.state_dict(), f"./weights/plot_mnist_{str(EPOCHS)}_twoLess.pt")
