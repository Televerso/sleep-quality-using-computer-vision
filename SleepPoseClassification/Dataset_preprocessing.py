import numpy as np
import cv2
import os

from basic_functions import basic_functions as bf
from matplotlib import pyplot as plt


path_test = "Dataset_RGB/Test"
path_train = "Dataset_RGB/Train"


path_leftlog = "left_log"
path_rightlog= "right_log"
path_leftprone = "prone_left"
path_rightprone = "prone_right"
path_supine = "supine"

path_pose = [path_supine, path_leftlog, path_rightlog, path_leftprone, path_rightprone]

def show_mage(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def process_image(path, angle, scale, thresh):
    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = bf.resize(image, 180, 135)
    image = bf.rotate(image,angle,scale)
    image = bf.threshold(image, thresh)
    image = bf.blur(image, 3, 2)
    return image


list_thresh = [100,120,140]
list_rot = [-20,-10,0,10,20]
list_scale = [0.8,0.9,1,1.1,1.2]


for pose in path_pose:
    path = os.path.join(path_train, pose)

    i = 0
    for item in os.listdir(path):

        for thresh in list_thresh:
            for rot in list_rot:
                for scale in list_scale:

                    path_item = os.path.join(path, item)
                    image = process_image(path_item,rot,scale,thresh)
                    image = bf.resize(image,64,64)
                    if np.sum(image) < 0.3*256*(image.shape[0]*image.shape[1]):
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(os.path.join(f"Dataset/Train/{pose}/{i}.png"), image)
                        i+=1

    print(i)
    print(pose)

for pose in path_pose:
    path = os.path.join(path_test, pose)

    i = 0
    for item in os.listdir(path):

        for thresh in list_thresh:
            for rot in list_rot:
                for scale in list_scale:

                    path_item = os.path.join(path, item)
                    image = process_image(path_item, rot, scale, thresh)
                    image = bf.resize(image,64,64)
                    if np.sum(image) < 0.3*256*(image.shape[0] * image.shape[1]):
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                        cv2.imwrite(os.path.join(f"Dataset/Test/{pose}/{i}.png"), image)
                        i += 1
    print(pose)