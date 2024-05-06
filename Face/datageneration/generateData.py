import sys
PATH_TO_DATA = "/media/hitchcock/data21/data"
sys.path.append(PATH_TO_DATA)
from img import dwGallery
import face_recognition
import hashlib
from PIL import Image
import os
import numpy as np



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
        print(image.shape)
        addVert = int(image.shape[1] * 0.1)
        addHor = int(image.shape[0] * 0.1)
        face_image = image[max(0, top - addVert):min(bottom + addVert, image.shape[0]),max(0, left - addHor):min(right + addHor, image.shape[1])]
        pil_image = Image.fromarray(face_image)
        pil_image.thumbnail((128, 128), Image.Resampling.LANCZOS)
        if pil_image.shape[0] == 128 and pil_image.shape[1] == 128:
            pil_image.save(path)
        else:
            os.remove(path)
        



def main():
    PassOneStack("", "m1")

if __name__=="__main__":main()