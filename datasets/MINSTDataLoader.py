from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
import os

class MNISTDataLoaderBase(Dataset):

    def __init__(self, train=True):
        def sortDataSet(dataSet):
            labels = torch.tensor([x[1] for x in dataSet])
            unique_labels, counts = torch.unique(labels, return_counts=True, sorted=True)
            self.num_labels = len(unique_labels)
            sort_labels = torch.argsort(labels)
            min_count = torch.min(counts)
            self.min_count = min_count
            assert min_count != 0
            if min_count % 2 != 0:
                min_count -= 1
            
            idx_list = []
            counter = 0
            for idx in sort_labels:
                # Check counter
                if counter == min_count and labels[sort_labels[idx - 1]] == labels[sort_labels[idx - 1]]:
                    continue
                elif counter == min_count:
                    counter = 0
                # Add next 2 idx
                indexes = sort_labels[idx], sort_labels[idx + 1]
                idx_list.append(indexes)
                counter += 2
            return torch.tensor(idx_list)
        
        self.data = datasets.MNIST(
            root="./data/MINST",
            train=train,
            download=True,
            transform=ToTensor()
        )
        if os.path.isfile("./data/MINST/{0}.pt".format("train" if train else "test")):
            self.set = torch.load("./data/MINST/{0}.pt".format("train" if train else "test"))
        else:
            self.set = sortDataSet(self.data)
            torch.save(self.set, "./data/MINST/{0}.pt".format("train" if train else "test"))

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.set[idx]
        return self.data[idx1] + self.data[idx2]

class MNISTDataLoaderMin(MNISTDataLoaderBase):

    def __init__(self, train=True):
        super(MNISTDataLoaderMin, self).__init__(train=train)
    
class MINSTDataLoaderMax(MNISTDataLoaderBase):

    def __init__(self, train=True):
        super(MINSTDataLoaderMax, self).__init__(train=train)
        shuffleSet = []
        for s in self.set:
            l1 = self.data[s[0]][1]
            while True:
                idx2 = torch.randint(0, len(self.set), (1,))
                c = int(self.set[idx2,1])
                d = self.data[c]
                if not l1 == d[1]:
                    break
            shuffleSet.append((s[0], idx2))
        self.set = torch.tensor(shuffleSet)
    
class MNISTDataLoaderMix(Dataset):

    def __init__(self, train=True):
        super(MNISTDataLoaderMix, self).__init__()
        self.minSet = MNISTDataLoaderMin(train=train)
        self.maxSet = MINSTDataLoaderMax(train=train)

    def __len__(self):
        return len(self.minSet) + len(self.maxSet)
    
    def __getitem__(self, idx):
        if idx < len(self.minSet):
            return self.minSet[idx]
        else:
            return self.maxSet[idx - len(self.minSet)]