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

def find_contours(img):
    (_, cnts, _) = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts
