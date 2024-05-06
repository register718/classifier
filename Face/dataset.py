import torch
import h5py
from torchvision import datasets
from torch.utils.data import Dataset
import scipy.io as io
import os
from PIL import Image
import numpy as np

class SiameseIMGDataset(Dataset):

    def __init__(self, pathData: str):
        hdf5Path = os.path.join(pathData, "database.hdf5")
        if not os.path.isfile(hdf5Path):
            self.createDatabase(pathData, hdf5Path)
        self.f = h5py.File(hdf5Path, mode='r')
        self.indexes = self.f["meta"]["idxs"]
        self.keyList = list(f.keys())
    
    def __del__(self):
        self.f.close()

    def createDatabase(self, dataPath, hdf5Path):
        with h5py.File(hdf5Path, mode="w") as f:
            indexes = []
            countPersons = 0
            for person in os.listdir(dataPath):
                personPath = os.path.join(dataPath, person)
                if not os.path.isdir(personPath):
                    continue
                grp = f.create_group(person)
                imgs = []
                countImg = 0
                for single in os.listdir(personPath):
                    singlePath = os.path.join(personPath, single)
                    pilImg = Image.open(singlePath)
                    imgArray = np.asarray(pilImg)
                    imgArray = np.expand_dims(imgArray, axis=0)
                    imgs.append(imgArray)
                    pilImg.close()
                    indexes.append((countPersons, countImg))
                    countImg += 1
                countPersons += 1
                npImg = np.concatenate(imgs)
                grp.create_dataset("imgs", data=npImg)
            idxGrp = f.create_group("meta")
            idexesNp = np.array(indexes)
            idxGrp.create_dataset("idxs", data=idexesNp)    

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idxs = self.indexes[idx]
        key = self.keyList[idxs[0]]
        anchorList = self.f[key]
        anchor = anchorList[idxs[1]]["imgs"]
        rIndx = np.random.random_integers(idxs[1], idxs[1] + len(anchorList) - 1) % len(anchorList)
        same = anchorList[rIndx]
        grpIndx = np.random.random_integers(idx, idx + len(self.keyList) - 1) % len(self.keyList)
        grpOther = self.f[self.keyList[grpIndx]]["imgs"]
        other =  np.random.choice(grpOther)
        return anchor, same, other

