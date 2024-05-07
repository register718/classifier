import sys
PATH_TO_DATA = "/media/hitchcock/data21/data"
sys.path.append(PATH_TO_DATA)
from img import dwGallery
import face_recognition
import hashlib
from PIL import Image
import os
import numpy as np


IMG_SIZE = 128
def PassOneStack(url, person):
    sp = os.path.join(PATH_TO_DATA, person)
    #dwGallery(url, sp)
    for name in os.listdir(sp):
        path = os.path.join(sp, name)
        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image)
        if not len(locations) == 1:
            os.remove(path)
            continue
        top, right, bottom, left = locations[0]

        addVert = int(image.shape[1] * 0.1)
        addHor = int(image.shape[0] * 0.1)

        face_image = image[max(0, top - addVert):min(bottom + addVert, image.shape[0]),max(0, left - addHor):min(right + addHor, image.shape[1])]
        pil_image = Image.fromarray(face_image)
        pil_image.thumbnail((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        np_image = np.asarray(pil_image)
        if np_image.shape[0] < IMG_SIZE:
            tmp = np.zeros((IMG_SIZE - np_image.shape[0], np_image.shape[1], 3))
            np_image = np.concatenate((tmp, np_image), axis=0)
        if np_image.shape[1] < IMG_SIZE:
            tmp = np.zeros((IMG_SIZE, IMG_SIZE - np_image.shape[1], 3))
            np_image = np.concatenate((tmp, np_image), axis=1)
        pil_image = Image.fromarray(np.uint8(np_image))
        pil_image.save(path)


def main(): 
    PassOneStack("", "m7")

if __name__=="__main__":main()