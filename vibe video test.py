import numpy as np
import os
import cv2
from PIL import Image

from ViBe import vibe
from basic_functions import basic_functions as bf

import time


class SimpleObjectDetection:
    def __init__(self, image):
        self.background_image = np.asarray(image)

    def detect(self, image, threshhold):
        image = np.asarray(image)
        object_image = np.abs(self.background_image - image)
        object_image[object_image < threshhold] = 0
        object_image[object_image >= threshhold] = 255

        return object_image


rootDir = r'Moving vid'
maskDir = r'/Vid masks'
maskDirSimple = r'/Vid masks simple'
origDir = r'/Original video'

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

dims = 360, 240

cap.set(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)

ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
scaled = cv2.resize(gray, dsize=(dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
print(scaled.shape[0], scaled.shape[1])
samples = vibe.initial_background(scaled, N)

cap.set(1,0)

orig_frame_list = list()


mask_list_vibe = list()
vibe_time_list = list()

ret, frame = cap.read()

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orig_frame_list.append(gray)
    scaled = cv2.resize(gray, dsize=(dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
    start = time.time()
    segMap, samples = vibe.vibe_detection(scaled, samples, _min, N, R)
    end = time.time()
    vibe_time_list.append(end - start)
    segMap = bf.blur(segMap, 7, 2)
    cv2.imshow('segMap', segMap)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break


    mask_list_vibe.append(segMap)
    ret, frame = cap.read()

cv2.destroyAllWindows()




cap.set(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)

ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
scaled = cv2.resize(gray, dsize=(dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
print(scaled.shape[0], scaled.shape[1])
scaled = cv2.medianBlur(scaled, 5)
simple_detector = SimpleObjectDetection(scaled)

cap.set(1,0)

mask_list_simple = list()
simple_time_list = list()

ret, frame = cap.read()

while ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, dsize=(dims[0], dims[1]), interpolation=cv2.INTER_NEAREST)
    scaled = cv2.medianBlur(scaled, 5)
    start = time.time()
    segMap = simple_detector.detect(scaled, 140)
    end = time.time()
    simple_time_list.append(end - start)
    # segMap = bf.blur(segMap, 7, 2)
    cv2.imshow('segMap', segMap)
    if cv2.waitKey(1) and 0xff == ord('q'):
        break

    mask_list_simple.append(segMap)
    ret, frame = cap.read()

cv2.destroyAllWindows()


img_num = 0
for mask in mask_list_vibe:
    outfile = str(rootDir + maskDir + f'/{img_num}.png')
    print(outfile)
    img_num += 1
    image = Image.fromarray(mask)
    image.save(outfile)

img_num = 0
for mask in mask_list_simple:
    outfile = str(rootDir + maskDirSimple + f'/{img_num}.png')
    print(outfile)
    img_num += 1
    image = Image.fromarray(mask)
    image.save(outfile)

img_num = 0
for frame in orig_frame_list:
    outfile = str(rootDir + origDir + f'/{img_num}.png')
    print(outfile)
    img_num += 1
    image = Image.fromarray(frame)
    image.save(outfile)

# for masks in os.listdir(maskDir):
#     path = os.path.join(maskDir, masks)
#     frame = cv2.imread(path)
#     cv2.imshow('frame', bf.blur(frame,7, 2))
#     cv2.waitKey(0)

cap.release()


print(np.sum(vibe_time_list)/len(vibe_time_list))
print(np.sum(simple_time_list)/len(simple_time_list))