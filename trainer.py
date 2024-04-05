import torch
from torch import optim
from torch.utils.data import DataLoader
from collections.abc import Iterable

class Trainer:

    def __init__(self, net: torch.nn.Module, train_data: Iterable, test_data: Iterable, EPOCHS=20):
        self.net = net
        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
        self.test_loader = DataLoader(test_data, shuffle=False)
        self.epochs = EPOCHS
    
    def train_Epoch(self):
            print("Starte Epoche")
            for x in self.train_loader:
                self.trainingStep(x)
    
    @torch.no_grad()
    def test_Epoch(self):
        liste = []
        for x in self.test_loader:
            res = self.testStep(x)
            if isinstance(res, list):
                liste += res
            else:
                liste.append(res)
        return liste

    def trainingStep(self, x):
        raise NotImplementedError()
    
    def testStep(self, x):
        raise NotImplementedError()
    
class MinTrainer(Trainer):

    def __init__(self, net: torch.nn.Module, train_set, test_set, lr=1e-3, EPOCHS=10):
        super().__init__(net, train_set, test_set, EPOCHS=EPOCHS)
        self.optimizerMin = optim.SGD(net.parameters(), maximize=False, lr=lr)

    
    def trainingStep(self, x):
        in1, l1, in2, l2 = x
        y1 = self.net(in1); y2 = self.net(in2)
        loss = torch.sum((y1-y2).pow(2))
        loss.backward()
        self.optimizerMin.step()
        self.optimizerMin.zero_grad()
    
    def testStep(self, x):
        in1, l1, in2, l2 = x
        y1 = self.net(in1); y2 = self.net(in2)
        loss = torch.sum((y1-y2).pow(2))
        return [(l1, loss, y1), (l2, loss, y2)]

class MaxTrainer(Trainer):

    def __init__(self, net: torch.nn.Module, train_set, test_set, lr=1e-3, EPOCHS=10):
        super().__init__(net, train_set, test_set, EPOCHS=EPOCHS)
        self.optimizerMin = optim.SGD(net.parameters(), maximize=True, lr=lr)

    def __lossF__(self, y1, y2):
        y2 = y2.detach()
        return (torch.sum((y1-y2).pow(2)) + 1e-8).pow(-1)
    
    def trainingStep(self, x):
        in1, l1, in2, l2 = x
        y1 = self.net(in1); y2 = self.net(in2)
        loss = self.__lossF__(y1, y2)
        if  loss > 1e6:
            #print("MAX", loss)
            loss.backward()
            self.optimizerMin.step()
            self.optimizerMin.zero_grad()
    
    def testStep(self, x):
        in1, l1, in2, l2 = x
        y1 = self.net(in1); y2 = self.net(in2)
        loss = self.__lossF__(y1, y2)
        if loss < 1e6:
            loss = 0
        return [(l1, loss, y1), (l2, loss, y2)]


class MinMaxTrainer(Trainer):

    def __init__(self, net: torch.nn.Module, train_set, test_set, lr=1e-3, EPOCHS=10):
        super().__init__(net, train_set, test_set, EPOCHS=EPOCHS)
        self.trainerMin = MinTrainer(net, train_set, test_set, lr=lr)
        self.trainerMax = MaxTrainer(net, train_set, test_set, lr=lr)
    
    def __step__(self, x, funcMax, funcMin):
        in1, l1, in2, l2 = x
        cp = l1 == l2
        x_eq = [x[i][cp] for i in range(len(x))]
        x_neq = [x[i][~cp] for i in range(len(x))]
        maxRes = []; minRes = []
        if not x_eq[0].shape[0] == 0:
            maxRes = funcMax(x_eq)
        if not x_neq[0].shape[0] == 0:
            minRes = funcMin(x_neq)
        return maxRes, minRes


    def trainingStep(self, x):
        self.__step__(x, self.trainerMax.trainingStep, self.trainerMin.trainingStep)

    def testStep(self, x):
        m1, m2 = self.__step__(x, self.trainerMax.testStep, self.trainerMin.testStep)
        return m1 + m2