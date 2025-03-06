import numpy as np
import os
import cv2
from PIL import Image
from ViBe import vibe
from basic_functions import basic_functions as bf

rootDir = r'Car moving 2'
maskDir = r'Car masks'

image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
image = cv2.imread(image_file, 0)

mask_file = os.path.join(maskDir, os.listdir(maskDir)[0])
mask = cv2.imread(image_file, 0)

for masks in os.listdir(maskDir):
    path = os.path.join(maskDir, masks)
    frame = cv2.imread(path)
    cv2.imshow('frame', bf.blur(frame,7, 2))
    cv2.waitKey(0)