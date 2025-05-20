import os
from tkinter import Image
from typing import Tuple

import cv2
import numpy as np
import time
import PIL

from FrameClass.FrameClass import Frame
from SleepPoseClassification.SleepPoseClassifyer import SleepPoseClassifyer
from ViBe import vibe
from basic_functions import basic_functions as bf

def save_frames_list(frames, path : str):
    """
    Сохраняет массив изображений frames в директории path
    """
    img_num = 0
    for frame in frames:
        outfile = os.path.join(path, f'{img_num}.png')
        img_num += 1
        cv2.imwrite(outfile, frame)

class SimpleObjectDetection:
    def __init__(self, image):
        self.background_image_ch1 = bf.blur(np.asarray(image[:,:,0]), 5, 1)
        self.background_image_ch2 = bf.blur(np.asarray(image[:,:,1]), 5, 1)
        self.background_image_ch3 = bf.blur(np.asarray(image[:,:,2]), 5, 1)

    def detect(self, image, threshold):
        image_ch1 = np.asarray(image[:,:,0])
        image_ch2 = np.asarray(image[:,:,1])
        image_ch3 = np.asarray(image[:,:,2])

        image_ch1 = bf.blur(image_ch1, 5, 1)
        image_ch2 = bf.blur(image_ch2, 5, 1)
        image_ch3 = bf.blur(image_ch3, 5, 1)

        image_diff_ch1 = np.abs(image_ch1.astype(int) - self.background_image_ch1.astype(int))
        image_diff_ch2 = np.abs(image_ch2.astype(int) - self.background_image_ch2.astype(int))
        image_diff_ch3 = np.abs(image_ch3.astype(int) - self.background_image_ch3.astype(int))


        image_diff_ch1[image_diff_ch1 < threshold] = 0
        image_diff_ch2[image_diff_ch2 < threshold] = 0
        image_diff_ch3[image_diff_ch3 < threshold] = 0
        image_diff_ch1[image_diff_ch1 >= threshold] = 255
        image_diff_ch2[image_diff_ch2 >= threshold] = 255
        image_diff_ch3[image_diff_ch3 >= threshold] = 255

        image_diff = image_diff_ch1 | image_diff_ch2 | image_diff_ch3

        return image_diff.astype('uint8')

class SleepTranscription:
    """
    Класс видеоряда сна, содержит методы для работы с кадрами видеоряда
    """
    def __init__(self, rootDir : str):
        self.cap = None
        self.curr_cap_frame = 0
        self.frame_list = list()
        self.simple_detector = None

        self.motion_images_list = list()
        self.motion_detection_list = list()
        self.pose_list = list()

        self.root = rootDir


    def open_videofile(self, filename : str) -> int:
        """
        Открывает видеофайл filename в заданной директории
        :param filename: Название видеофайла
        :return: Код результата открытия
        """
        path_to_file = os.path.join(self.root, filename)
        self.cap = cv2.VideoCapture(path_to_file)

        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return -1
        else:
            print("Video file opened successfully!")
            return 0


    def add_next_frame(self,  dims : Tuple[int,int]):
        """
        Считывает текущий кадр и переходит к следующему, заполняя внутренний список
        :return: Считанный кадр; -1 при ошибке чтения
        """
        ret, frame = self.cap.read()
        if ret:
            frame = bf.resize(frame, dims[0], dims[1])

            frame_obj = Frame(frame, self.curr_cap_frame)
            self.curr_cap_frame += 1
            self.frame_list.append(frame_obj)
            return frame_obj
        else: return -1


    def set_cap_to_n_frame(self, n : int):
        """
        Устанавливает позицию текущего кадра
        :param n: Номер кадра
        """
        self.cap.set(1, n)
        self.curr_cap_frame = n


    def set_cap_to_first_frame(self):
        """
        Устанавливает позицию текущего кадра на первый кадр видеоряда
        """
        self.cap.set(1, 0)
        self.curr_cap_frame = 0


    def get_total_cap_framecount(self):
        """
        :return: Возвращает общее количество кадров видеоряда
        """
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)


    def set_cap_to_last_frame(self):
        """
        Устанавливает позицию текущего кадра на последний кадр видеоряда
        """
        self.cap.set(1, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.curr_cap_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1


    def read_cap_frames(self, dims : Tuple[int,int], gap : int = 1, rotate_param = 0):
        """
        Поочередно считывает кадры видеоряда, заполняя внутренний список
        :param gap: Количество пропускаемых кадров, которые не будут записаны
        :param rotate_param: -1, 0, 1, 2 - вращение кадров на 90*rotate градусов
        """
        rotate = None
        if rotate_param == -1:
            rotate = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotate_param == 1:
            rotate = cv2.ROTATE_90_CLOCKWISE
        elif rotate_param == 2:
            rotate = cv2.ROTATE_180

        ret, frame = self.cap.read()
        i = 0
        while ret:
            if i%gap == 0:
                frame = bf.resize(frame, dims[0], dims[1])

                frame_obj = None
                if rotate is None:
                    frame_obj = Frame(frame, self.curr_cap_frame)
                else:
                    frame_obj = Frame(cv2.rotate(frame, rotate), self.curr_cap_frame)

                self.frame_list.append(frame_obj)

            self.curr_cap_frame += 1
            i += 1
            ret, frame = self.cap.read()

    def save_frames(self, path : str):
        """
        Сохраняет все кадры в указанной директории
        :param path: Относительный адрес директории
        """
        img_num = 0
        for frame in self.frame_list:
            outfile = os.path.join(self.root, f'{path}/{img_num}.png')
            img_num += 1
            cv2.imwrite(outfile, frame.image)


    def save_masks(self, path : str):
        """
        Сохраняет все маски объектов на кадрах в указанной директории
        :param path: Относительный адрес директории
        """
        img_num = 0
        for frame in self.frame_list:
            outfile = os.path.join(self.root, f'{path}/{img_num}.png')
            img_num += 1
            cv2.imwrite(outfile, frame.mask)


    def close_video(self):
        """
        Закрывает объект capture видеоряда.
        После вызова взаимодействие с видеорядом сможет осуществляться только через список сохраненных кадров
        """
        self.cap.release()
        self.cap = None


    def resize_frames(self, dims : Tuple[int,int]) -> list:
        """
        Преобразует все записанные кадры к указанному размеру
        :param dims: Новые размеры крдров видеоряда
        :return: Список сохраненных кадров
        """
        for frame in self.frame_list:
            frame.resize(dims)
        return self.frame_list


    def blur_frames(self) -> list:
        """
        Производит медианную фильтрацию всех записанных кадров
        :return: Список сохраненных кадров
        """
        for frame in self.frame_list:
            frame.median_blur()
        return self.frame_list


    def do_ViBe_algorithm(self, params : tuple = (20,40,2,16), n_of_init_frame : int = 0) -> list:
        """
        Применяет алгоритм обнаружения движений ViBe ко всем сохраненным кадрам видеоряда
        :param params: кортеж параметров алгоритма (N, R, _min, phai)
        :param n_of_init_frame: Номер исходного кадра
        :return: Список сохраненных кадров с масками объектов
        """
        N = params[0]
        R = params[1]
        _min = params[2]
        phai = params[3]


        samples_ch1 = vibe.initial_background(self.frame_list[n_of_init_frame].image[:,:,0], N)
        samples_ch2 = vibe.initial_background(self.frame_list[n_of_init_frame].image[:,:,1], N)
        samples_ch3 = vibe.initial_background(self.frame_list[n_of_init_frame].image[:,:,2], N)
        for i in range(len(self.frame_list)):
            segMap_ch1, samples_ch1 = vibe.vibe_detection(self.frame_list[i].image[:, : ,0], samples_ch1, _min, N, R)
            segMap_ch2, samples_ch2 = vibe.vibe_detection(self.frame_list[i].image[:, :, 1], samples_ch2, _min, N, R)
            segMap_ch3, samples_ch3 = vibe.vibe_detection(self.frame_list[i].image[:, :, 2], samples_ch3, _min, N, R)

            segMap = segMap_ch1 | segMap_ch2 | segMap_ch3

            self.frame_list[i].add_m_mask(segMap)

        return self.frame_list


    def detect_object_frames(self, threshold_obj : float = 0.02) -> list:
        """
        Помечает кадры, на которых был обнаружен объект
        :param threshold_obj: Доля пикселей, занимаемых объектом
        :return: bool-список соответствующий сохраненным кадрам
        """
        new_pixel_thresh = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1] * threshold_obj)
        self.simple_detector = SimpleObjectDetection(self.frame_list[0].image)

        for i in range(len(self.frame_list)):

            self.frame_list[i].add_mask(self.simple_detector.detect(self.frame_list[i].image, threshold=40))
            self.frame_list[i].check_object_presence(new_pixel_thresh)

        return [frame.object_present for frame in self.frame_list]


    def detect_motion(self, n=5) -> list:
        """
        Определяет движения объекта на видеоряде
        :param n: Количество кадров, на протяжении которых производится определение движений
        :return:
        """
        pixel_count = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1])

        self.motion_images_list = list()
        self.motion_detection_list = list()

        for i in range(len(self.frame_list)):
            n_iter = n
            if i-n_iter < 0:
                n_iter = i
            frame = np.zeros_like(self.frame_list[i].m_mask)
            for j in range(0, n_iter):
                frame |= self.frame_list[i].m_mask ^ self.frame_list[i - j].m_mask
            self.motion_images_list.append(frame)
            self.motion_detection_list.append(np.sum(frame)/pixel_count)
        return self.motion_detection_list


    def classify_poses(self):
        sleep_model = SleepPoseClassifyer()
        obj_masks = list()
        for frame in self.frame_list:
            obj_masks.append(bf.get_64pix_mask(frame.m_mask))
        self.pose_list = sleep_model.batch_classify(obj_masks)

        obj_present = self.detect_object_frames()
        for pose in range(len(self.pose_list)):
            if not obj_present[pose]:
                self.pose_list[pose] = -1
        # save_frames_list(obj_masks, r"C:\Users\dboga\PycharmProjects\sleep quality using computer vision\Moving vid\Test_64")
        return self.pose_list


    def prosess_frames(self, dims : Tuple[int,int], thresh_object : float, n : int, thresh_motion : float) ->\
            'SleepTranscription':
        """
        Производит полную обработку видеоряда, последовательно вызывая методы класса
        :param dims: Размерность кадра, к которым должен быть приведен видеоряд
        :param thresh_object: Доля кадра, которую должен занимать объект для его обнаружения
        :param n: Глубина поиска движений в кадрах
        :param thresh_motion: Доля кадра, которую должно занять движение для его фиксации
        :return:
        """
        self.detect_object_frames(thresh_object)
        self.do_ViBe_algorithm()
        self.blur_frames()
        self.detect_motion(n)
        self.classify_poses()

        return self


    def get_motion_frames(self, threshhold : float) -> list:
        """
        :return: возвращает список кадров, на которых было определено движение объекта
        """
        frames = list()
        for i in range(len(self.motion_detection_list)):
            if (self.motion_detection_list[i] > threshhold) & self.frame_list[i].object_present:
                frames.append(self.frame_list[i])
        return frames

    def prosess_frames_timed(self, thresh_object : float, n : int, thresh_motion : float) ->\
            'SleepTranscription':
        """
        Производит полную обработку видеоряда, последовательно вызывая методы класса
        :param dims: Размерность кадра, к которым должен быть приведен видеоряд
        :param thresh_object: Доля кадра, которую должен занимать объект для его обнаружения
        :param n: Глубина поиска движений в кадрах
        :param thresh_motion: Доля кадра, которую должно занять движение для его фиксации
        :return:
        """
        start = time.time()
        self.detect_object_frames(thresh_object)
        end = time.time()
        print("Preprocessing time = ", (end - start)/len(self.frame_list))

        start = time.time()
        self.do_ViBe_algorithm()
        end = time.time()
        print("ViBe time = ", (end - start)/len(self.frame_list))

        start = time.time()
        self.blur_frames()
        self.detect_motion(n)
        end = time.time()
        print("Masks processing time = ", (end - start) / len(self.frame_list))

        start = time.time()
        self.classify_poses()
        end = time.time()
        print("Pose classification time = ", (end - start) / len(self.frame_list))

        return self