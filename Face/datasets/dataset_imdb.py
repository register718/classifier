import h5py
from torch.utils.data import Dataset
import scipy.io as io
import os
from PIL import Image
import numpy as np
from os.path import join as j
from tqdm import tqdm
import torch


def crop_to_square(image, top, bottom, left, right):
    def smallest_square_region(top, bottom, left, right):
        # Berechne die Höhe und Breite des Bereichs
        height = bottom - top
        width = right - left
        
        # Bestimme die Seitenlänge des Quadrats
        size = max(height, width)
        
        # Berechne die Koordinaten des Quadrats
        square_top = top
        square_bottom = top + size
        square_left = left
        square_right = left + size
        # move vert
        moveVert = ((square_bottom - square_top) - (bottom - top)) // 2 # optimal
        moveVert = min(moveVert, square_top)  # maximal bis an den rand
        square_top, square_bottom = square_top - moveVert, square_bottom - moveVert
        # move horz
        moveHorz = ((square_right - square_left) - (right - left)) // 2
        moveHorz = min(moveHorz, square_left)
        square_left, square_right = square_left - moveHorz, square_right - moveHorz
        
        return square_top, square_bottom, square_left, square_right
    height, width, _ = image.shape
    top, bottom, left, right = smallest_square_region(top, bottom, left, right)
    
    # Überprüfe, ob ein Quadrat innerhalb des Bereichs möglich ist
    if top < 0 or left < 0:
        return np.NaN
    if bottom > height:
        diff = bottom - height
        if diff > top:
            return np.NaN
        top, bottom = top - diff, bottom - diff
    if right > width:
        diff = right - width
        if diff > left:
            return np.NaN
        left, right = left - diff, right - diff
    
    # Schneide das Quadrat aus dem Bild aus
    cropped_img_array = image[top:bottom, left:right]
    return cropped_img_array

def crop_to_square_force(image):
    width, height, _ = image.shape
    
    # Bestimme die Größe des Quadrats
    size = min(width, height)
    
    # Berechne die Koordinaten des Ausschnitts
    left = (width - size) // 2
    top = (height - size) // 2
    right = (width + size) // 2
    bottom = (height + size) // 2
    
    # Schneide das Quadrat aus dem Bild aus
    cropped_image = image[top:bottom, left:right]
    return cropped_image

class SiameseIMDBDataset(Dataset):

    def __init__(self, dataPath):
        h5Path = j(dataPath, "data.hdf5")
        if not os.path.isfile(h5Path):
            print("Create hdf5 Database")
            self.createHDF5(dataPath, h5Path)
        self.f = h5py.File(h5Path, mode="r")
        ky = self.f["meta"].keys()
        self.keys = []
        for k in ky:
            self.keys += [(k, x) for x in self.f["meta"][k]]
        self.num_grps = len(self.f["meta"])
        
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        grp, idx = self.keys[index]
        idxInt = int(idx)
        idxSame = idxInt
        while idxInt == idxSame:
            idxSame = np.random.randint(0, len(self.f["meta"][grp].keys()))
        idxOther = index
        while self.keys[idxOther][0] == self.keys[index][0]:
            idxOther = np.random.randint(0, self.num_grps)
        grpOther, idxOther = self.keys[idxOther]
        anchorPath = self.f["meta"][grp][idx].asstr()[()]
        samePath = self.f["meta"][grp][str(idxSame)].asstr()[()]
        otherPath = self.f["meta"][grpOther][idxOther].asstr()[()]
        anchor = torch.from_numpy(self.f["images"][anchorPath][:])
        same = torch.from_numpy(self.f["images"][samePath][:])
        other = torch.from_numpy(self.f["images"][otherPath][:])
        return anchor, same, other
        
        
        
    def createHDF5(self, dataPath, h5Path):
        IMG_SIZE = 220
        meta = io.loadmat(j(dataPath, "imdb.mat"))["imdb"]
        length = meta["name"][0,0].shape[1]
        with h5py.File(h5Path, mode="w") as f:
            imgs = f.create_group("images")
            mt = f.create_group("meta")
            for i in tqdm(range(length), desc="Creating Database"):
                second_face = meta["second_face_score"][0,0][0,i]
                if not np.isnan(second_face):
                    continue
                first_face = meta["face_score"][0,0][0,i]
                if first_face == np.Inf:
                    continue
                name = str(meta["name"][0,0][0,i][0])
                #print(name)
                full_path: str = str(meta["full_path"][0,0][0,i][0])
                ## LOAD IMG ## 
                #print(full_path)
                img_path = os.path.join(dataPath, "imdb", full_path)
                if not os.path.isfile(img_path):
                    print("ERROR: File not found", img_path)
                    continue
                try:
                    fimage = Image.open(img_path)
                    image = np.asarray(fimage)
                    fimage.close()
                except Exception as e:
                    print(e)
                    continue
                ## Bild zu klein
                if image.shape[0] < IMG_SIZE or image.shape[1] < IMG_SIZE or len(image.shape) == 2:
                    #print("Bild zu klein")
                    continue
                
                if name not in mt:
                    nGrp = mt.create_group(name)
                else:
                    nGrp = mt[name]
                n = len(nGrp.keys())
                nGrp.create_dataset(str(n), data=full_path)
                    
                ## Add Image ##
                f_loc: np.array = meta["face_location"][0,0][0,i]
                
                #plt.imshow(image)
                #plt.show()
                fimage.close()
                right, top, left, bottom = f_loc[0].round().astype(np.int32)
                addVert = int(image.shape[1] * 0.1)
                addHor = int(image.shape[0] * 0.1)
                ## Check size ##
                top = max(0, top - addVert)
                bottom = min(bottom + addVert, image.shape[0])
                left = max(0, left - addHor)
                right = min(right + addHor, image.shape[1])

                # IMG ZUSCHNEIDEN
                face_image = crop_to_square(image, top, bottom, left, right)
                if np.isnan(face_image).any():
                    #face_image = crop_to_square_force(image)
                    #right, top, left, bottom = f_loc[0].round().astype(np.int32)
                    #if face_image.shape[0] < bottom and face_image.shape[1] < left:
                    #    continue
                    #plt.imshow(face_image)
                    #plt.show()
                    continue
                pil_image = Image.fromarray(face_image)
                pil_image.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                #plt.imshow(pil_image)
                #plt.show()
                np_image = np.asarray(pil_image)
                #plt.imshow(np_image)
                #plt.show()
                
                #print("IMG SHAPE", np_image.shape)
                #plt.imshow(np_image)
                #plt.show()
                np_image = np_image / 256
                imgs.create_dataset(full_path, data=np_image)
            ## Check auf Einzelpersonen ##
            for i in f["meta"].keys():
                if len(f["meta"][i]) == 1:
                    del f["meta"][i]
                    
                
        
        
            


                
