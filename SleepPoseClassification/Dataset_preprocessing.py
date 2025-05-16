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

# path_pose = [path_supine, path_leftlog, path_rightlog, path_leftprone, path_rightprone]
path_pose = [path_supine, path_leftlog, path_rightlog]

def show_image(image):
    plt.plot()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def process_image(path, angle, scale, thresh):
    image = cv2.imread(path) # Считывает изображение

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Приводит изображение к отенкам серого
    image = bf.resize(image, 180*2, 135*2) # Подгоняет изображения к общему разрешению

    coef = 70/(np.sum(image)/(image.shape[0]*image.shape[1]))
    image = (image*coef).astype('uint8') # Нормализуем среднюю яркость к 70

    image[-25:-1,0:60] = 0 # Удаляет водяной знак на изображениях датасета
    image = bf.rotate(image,angle,scale) # Вращает и изменяет масштаб изображения в соответствии с переданными значениями
    image = bf.threshold(image, thresh) # Приводит изображение к бинарному в соответствии с переданным значением
    image = bf.blur(image, 7, 2) # Применяет медианный фильтр 3х3 К изображению
    image = bf.get_64pix_mask(image) # Приводит изображение к размеру 64х64, используемому при обучении сети

    return image


# show_image(process_image(os.path.join(os.path.join(path_test,path_supine),os.listdir(os.path.join(path_test,path_supine))[0]), 0, 1, 120))

for pose in path_pose:
    path = os.path.join(path_train, pose)

    i = 0
    for item in os.listdir(path):
        th_low = 105
        th_high = 135
        for var in range(0,10):
            path_item = os.path.join(path, item)
            image = process_image(path_item,np.random.randint(0,40)-20,1,np.random.randint(th_low,th_high))

            if np.sum(image) < 0.35*3*256*(image.shape[0]*image.shape[1]) and np.sum(image) > 0.1*3*256*(image.shape[0]*image.shape[1]):

                cv2.imwrite(os.path.join(f"Dataset/Train/{pose}/{i}.png"), image)
                i+=1

    print(i)
    print(pose)

for pose in path_pose:
    path = os.path.join(path_test, pose)

    i = 0
    for item in os.listdir(path):

        for var in range(0,10):
            path_item = os.path.join(path, item)
            image = process_image(path_item,np.random.randint(0,30)-15,1,np.random.randint(90,130))

            if np.sum(image) < 0.35*3*256*(image.shape[0]*image.shape[1]) and np.sum(image) > 0.1*3*256*(image.shape[0]*image.shape[1]):

                cv2.imwrite(os.path.join(f"Dataset/Test/{pose}/{i}.png"), image)
                i += 1
    print(pose)