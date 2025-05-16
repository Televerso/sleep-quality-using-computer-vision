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
    max_area_img = np.argsort(-stats[1:, 4]) + 1
    x, y, w, h, s = stats[max_area_img[0]]
    cx, cy = x + w // 2, y + h // 2

    for i in range(1, 3):
        xi, yi, wi, hi, si = stats[max_area_img[i]]
        coef = (si) / s
        cxi, cyi = xi + wi // 2, yi + hi // 2
        signxi = np.sign(cxi - cx)
        signyi = np.sign(cyi - cy)

        x_new = int(x + signxi * (xi // 2) * coef)
        y_new = int(y + signyi * (yi // 2) * coef)
        w_new = int(w + signxi * (wi // 2) * coef)
        h_new = int(h + signyi * (hi // 2) * coef)

        if x_new<x: x = x_new
        if y_new<y: y = y_new
        if w_new>w: w = w_new
        if h_new>h: h = h_new

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