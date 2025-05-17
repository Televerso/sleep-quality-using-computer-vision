import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from basic_functions import basic_functions as bf


def show_image(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

image0 = cv2.imread("Sleeping vid 1/Test_class_f/0.png")
image1 = cv2.imread("Sleeping vid 1/Test_class_f/1000.png")

show_image(image0)

show_image(image1)

image_diff = image1.astype(int)-image0.astype(int)
image_diff[image_diff[:,:]<0] = 0

show_image(image_diff)
print(np.abs(image1-image0).max())

image0 = bf.blur(image0, 5, 1)
image1 = bf.blur(image1, 5, 1)

image_diff = image1.astype(int)-image0.astype(int)
image_diff[image_diff[:,:]<0] = 0
show_image(image_diff)