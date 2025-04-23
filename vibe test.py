import numpy as np
import os
import cv2
from PIL import Image
from ViBe import vibe
from basic_functions import basic_functions as bf

rootDir = r'Car moving 2'

image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
image = cv2.imread(image_file, 0)

N = 20
R = 20
_min = 2
phai = 16
print(image.shape)

samples = vibe.initial_background(image, N)



mask_list = list()
for lists in os.listdir(rootDir):
    path = os.path.join(rootDir, lists)
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segMap, samples = vibe.vibe_detection(gray, samples, _min, N, R)
    mask_list.append(segMap)
    cv2.imshow('segMap', segMap)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break
cv2.destroyAllWindows()

img_num = 0
for mask in mask_list:
    outfile = os.path.join(rootDir, f'{img_num}.png')
    print(outfile)
    img_num += 1
    image = Image.fromarray(mask)
    image.save(outfile)
