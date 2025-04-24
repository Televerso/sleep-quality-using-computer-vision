import os
from typing import Tuple

import cv2
import numpy as np

from FrameClass.FrameClass import Frame
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


class SleepTranscription:
    """
    Класс видеоряда сна, содержит методы для работы с кадрами видеоряда
    """
    def __init__(self, rootDir : str):
        self.cap = None
        self.curr_cap_frame = 0
        self.frame_list = list()

        self.motion_masks_list = list()
        self.motion_detection_list = list()

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


    def add_next_frame(self):
        """
        Считывает текущий кадр и переходит к следующему, заполняя внутренний список
        :return: Считанный кадр; -1 при ошибке чтения
        """
        ret, frame = self.cap.read()
        if ret:
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


    def read_cap_frames(self, gap : int = 1):
        """
        Поочередно считывает кадры видеоряда, заполняя внутренний список
        :param gap: Количество пропускаемых кадров, которые не будут записаны
        """
        ret, frame = self.cap.read()
        i = 0
        while ret:
            if i%gap == 0:
                frame_obj = Frame(frame, self.curr_cap_frame)
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


    def grayscale_frames(self) -> list:
        """
        Приводит все записанные кадры к оттенкам серого
        :return: Список сохраненных кадров
        """
        for frame in self.frame_list:
            frame.to_grayscale()
        return self.frame_list


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


    def do_ViBe_algorithm(self, params : tuple = (20,20,2,16), n_of_init_frame : int = 0) -> list:
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

        samples = vibe.initial_background(self.frame_list[n_of_init_frame].image, N)
        for frame in self.frame_list:
            segMap, samples = vibe.vibe_detection(frame.image, samples, _min, N, R)
            frame.add_mask(segMap)
        return self.frame_list


    def detect_object_frames(self, threshold : float = 0.02) -> list:
        """
        Помечает кадры, на которых был обнаружен объект
        :param threshold: Доля пикселей, занимаемых объектом
        :return: bool-список соответствующий сохраненным кадрам
        """
        new_pixel_thresh = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1] * threshold)

        for frame in self.frame_list:
            frame.check_object_presence(new_pixel_thresh)

        return [frame.object_present for frame in self.frame_list]


    def detect_motion(self, n=5, threshold : float = 0.01) -> list:
        """
        Определяет движения объекта на видеоряде
        :param n: Количество кадров, на протяжении которых производится определение движений
        :param threshold: Доля изменившихся пикселей на кадре, при привышении которой будет зафиксировано движение
        :return:
        """
        new_pixel_thresh = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1] * threshold)
        self.motion_masks_list = list()
        self.motion_detection_list = list()

        for i in range(len(self.frame_list)):
            n_iter = n
            if i-n_iter < 0:
                n_iter = i
            frame = np.zeros_like(self.frame_list[i].mask)
            for j in range(0, n_iter):
                frame |= self.frame_list[i-j].mask ^ self.frame_list[i].mask
            self.motion_masks_list.append(frame)
            self.motion_detection_list.append(np.sum(frame) > new_pixel_thresh)
        return self.motion_detection_list


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
        self.grayscale_frames()
        self.resize_frames(dims)
        self.do_ViBe_algorithm()
        self.blur_frames()
        self.detect_object_frames(thresh_object)
        self.detect_motion(n, thresh_motion)
        return self


    def get_motion_frames(self) -> list:
        """
        :return: возвращает список кадров, на которых было определено движение объекта
        """
        frames = list()
        for i in range(len(self.motion_detection_list)):
            if self.motion_detection_list[i] & self.frame_list[i].object_present:
                frames.append(self.frame_list[i])
        return frames