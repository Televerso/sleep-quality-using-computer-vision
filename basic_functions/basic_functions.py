import cv2
import numpy as np

def resize(img, h, w):
    res_img = cv2.resize(img, (w, h), cv2.INTER_AREA)
    return res_img

def shift(img, x=0, y=0):
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(img, translation_matrix, (w, h))
    return dst

def rotate(img, angle, scale=1.0):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    return rotated

def blur(img, ksize, mode):
    if mode == "average" or mode == 0:
        return cv2.blur(img, (ksize,ksize))
    elif mode == "gaussian" or mode == "gaussian_blur" or mode == 1:
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif mode == "median" or mode == "median_blur" or mode == 2:
        return cv2.medianBlur(img, ksize)

def threshold(img, thresh):
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]


def get_64pix_mask(image):
    _, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    max_area_img = np.argsort(-stats[1:, 4]) + 1

    if len(max_area_img) == 0 or len(max_area_img) > 5:
        image_64 = np.zeros((64, 64, 3), dtype=np.uint8)
        return image_64

    x, y, w, h, s = stats[max_area_img[0]]
    cx, cy = x + w // 2, y + h // 2
    w,h = x+w, y+h

    for i in max_area_img[1:3]:
        xi, yi, wi, hi, si = stats[i]
        coef = (si) / s
        cxi, cyi = xi + wi // 2, yi + hi // 2
        wi,hi = xi+wi,yi+hi
        signxi = np.sign(cxi - cx)
        signyi = np.sign(cyi - cy)

        x_new = int(x + signxi * (xi // 2) * coef)
        y_new = int(y + signyi * (yi // 2) * coef)
        w_new = int(w + signxi * (wi // 2) * coef)
        h_new = int(h + signyi * (hi // 2) * coef)

        if x_new < x: x = x_new
        if y_new < y: y = y_new
        if w_new > w: w = w_new
        if h_new > h: h = h_new


    margin = (w+h)//5

    lx = 0 if x-margin<0 else x-margin
    ly = 0 if y-margin<0 else y-margin
    rx = image.shape[1] - 1 if w+margin>image.shape[1] else w+margin
    ry = image.shape[0] - 1 if h+margin>image.shape[0] else h+margin

    image_64 = np.zeros((64, 64, 3), dtype=np.uint8)
    image_64 = cv2.resize(image[ly:ry,lx:rx],(64,64),interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(image_64, cv2.COLOR_GRAY2RGB)