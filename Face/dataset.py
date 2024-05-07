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
        self.indexes = self.f.get("meta")["idxs"]
        self.keyList = list(self.f.keys())
    
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
        anchorList = self.f.get(key)
        anchor = anchorList["imgs"][idxs[1]]
        rIndx = np.random.random_integers(idxs[1], idxs[1] + len(anchorList) - 1) % len(anchorList)
        same = anchorList["imgs"][rIndx]
        grpIndx = np.random.random_integers(idx, idx + len(self.keyList) - 1) % len(self.keyList)
        grpOther = self.f.get(self.keyList[grpIndx])["imgs"]
        other =  grpOther[np.random.randint(0, len(grpOther))]
        return torch.from_numpy(anchor), torch.from_numpy(same), torch.from_numpy(other)

