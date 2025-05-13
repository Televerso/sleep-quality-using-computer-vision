import cv2
import matplotlib.pyplot as plt
import numpy as np
import basic_functions.basic_functions as bf

def show_image(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def get_64pix_mask(image):
    _, _, stats, _ = cv2.connectedComponentsWithStats(image[:, :, 0], connectivity=4)
    max_area_img = np.argmax(stats[1:, 4]) + 1
    x, y, w, h, _ = stats[max_area_img]
    cx, cy = x + w // 2, y + h // 2

    margin = (w+h)//6

    lx = 0 if x-margin<0 else x-margin
    ly = 0 if y-margin<0 else y-margin
    rx = image.shape[1] - 1 if x+w+margin>image.shape[1] else x+w+margin
    ry = image.shape[0] - 1 if y+h+margin>image.shape[0] else y+h+margin

    image_64 = np.zeros((64, 64, 3), dtype=np.uint8)
    image_64 = cv2.resize(image[ly:ry,lx:rx],(64,64),interpolation=cv2.INTER_NEAREST)

    return image_64

image2 = cv2.imread("Car masks/mask (7).png")
image1 = cv2.imread("Moving vid/Test_class_m/1.png")

show_image(image2)
show_image(get_64pix_mask(image2))

show_image(image1)
show_image(get_64pix_mask(image1))