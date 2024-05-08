import torch
import h5py
from torchvision import datasets
from torch.utils.data import Dataset
import scipy.io as io
import os
from PIL import Image
import numpy as np
from os.path import join as j

class SiameseIMDBDataset(Dataset):

    def __init__(self, dataPath):
        h5Path = j(dataPath, "data.hdf5")
        if not os.path.isfile(h5Path):
            print("Create hdf5 Database")
            self.createHDF5(dataPath, h5Path)
        
    def createHDF5(self, dataPath, h5Path):
        meta = io.loadmat(j(dataPath, "imdb.mat"))["imdb"]
        length = meta["name"][0,0].shape[1]
        with h5py.File(h5Path, mode="w") as f:
            imgs = f.create_group("images")
            mt = f.create_group("meta")
            for i in range(length):
                second_face = meta["second_face_score"][0,0][0,i]
                if not second_face == np.NAN:
                    continue
                first_face = meta["face_score"][0,0][0,i]
                if first_face == np.Inf:
                    continue
                name = meta["name"][0,0][0,i]
                full_path = meta["full_path"][0,0][0,i:i+1]
                if name in mt:
                    mt.create_dataset(name, data=full_path)
                else:
                    d = mt[name]
                    d_neu = np.concatenate((d, full_path), axis=0)
                    mt[name][...] = d_neu
        
        
            


                
