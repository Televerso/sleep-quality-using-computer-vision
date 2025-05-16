import cv2
import matplotlib.pyplot as plt
import numpy as np
import basic_functions.basic_functions as bf
import PIL

def show_image(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


image1 = cv2.imread("SleepPoseClassification/Dataset_RGB/Train/right_log/awais (1).jpg")
image2 = cv2.imread("SleepPoseClassification/Dataset_RGB/Train/right_log/chainese1 (1).jpg")

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

coef1 = 70/(np.sum(image1)/(image1.shape[0]*image1.shape[1]))
coef2 = 70/(np.sum(image2)/(image2.shape[0]*image2.shape[1]))
print(coef1, coef2)

print(np.sum(image1)/(image1.shape[0]*image1.shape[1]))
image1 = image1*coef1
print(np.sum(image1)/(image1.shape[0]*image1.shape[1]))

print(np.sum(image2)/(image2.shape[0]*image2.shape[1]))
image2 = image2*coef2
print(np.sum(image2)/(image2.shape[0]*image2.shape[1]))

show_image(image1)
show_image(np.dot(image1,coef1))
show_image(image2)
show_image(np.dot(image2,coef2))
