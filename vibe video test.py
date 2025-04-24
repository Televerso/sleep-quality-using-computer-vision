import numpy as np
import os
import cv2
from PIL import Image

from FrameClass.FrameClass import *
from ViBe import vibe
from basic_functions import basic_functions as bf

rootDir = r'Moving vid'
maskDir = r'Vid masks'

# image_file = os.path.join(rootDir, os.listdir(rootDir)[0])
# image = cv2.imread(image_file, 0)
#
# mask_file = os.path.join(maskDir, os.listdir(maskDir)[0])
# mask = cv2.imread(image_file, 0)

cap = cv2.VideoCapture(rootDir+r'/video.mp4')
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully!")


N = 20
R = 20
_min = 2
phai = 16

cap.set(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)

ret, frame = cap.read()
frame = Frame(frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)

gray = frame.to_grayscale()
scaled = gray.resize((360,240))
print(scaled.image_size)
samples = vibe.initial_background(scaled.image, N)

cap.set(1,0)

mask_list = list()
while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, dsize=(360, 240), interpolation=cv2.INTER_NEAREST)
    segMap, samples = vibe.vibe_detection(scaled, samples, _min, N, R)
    segMap = bf.blur(segMap, 7, 2)
    cv2.imshow('segMap', segMap)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break


    mask_list.append(segMap)
    ret, frame = cap.read()




cv2.destroyAllWindows()

img_num = 0
for mask in mask_list:
    outfile = os.path.join(maskDir, f'/{img_num}.png')
    print(outfile)
    img_num += 1
    image = Image.fromarray(mask)
    image.save(outfile)


# for masks in os.listdir(maskDir):
#     path = os.path.join(maskDir, masks)
#     frame = cv2.imread(path)
#     cv2.imshow('frame', bf.blur(frame,7, 2))
#     cv2.waitKey(0)

cap.release()