from enum import Flag, auto
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray

from ViBe import vibe
from basic_functions import basic_functions as bf


class FrameStatus(Flag):
    """
    Класс с флагами состояний кадра:
    RGB, GRAY, SHRINKED, MASKED, BLURRED, OBJECT_CHECKED
    Используется во внутренних функциях
    """
    RGB = auto()
    SHRINKED = auto()
    MASKED = auto()
    M_MASKED = auto()
    BLURRED = auto()
    OBJECT_CHECKED = auto()

class Frame:
    """
    Класс отдельного кадра видео. Содержит расширенную информацию помимо изображения из видео.
    """
    def __init__(self, frame : ndarray, fr_num : int):
        self._object_present = False
        self._m_mask = None
        self._mask = None
        self._image = frame
        self._fr_num = fr_num
        self.__status = FrameStatus.RGB

    def __copy__(self) -> 'Frame':
        new_frame =  Frame(self.image, self.fr_num)
        new_frame.__status = self.__status
        return new_frame

    def __deepcopy__(self, memo) -> 'Frame':
        new_frame = Frame(self.image.copy(), self.fr_num)
        new_frame.__status = self.__status
        return new_frame

    def __str__(self):
        obj_str = "Frame N:"
        obj_str += str(self.fr_num)

        obj_str += ", Status: ("
        if self.__status & FrameStatus.RGB:
            obj_str += " RGB"
        elif self.__status & FrameStatus.SHRINKED:
            obj_str += " SHRINKED"
        elif self.__status & FrameStatus.BLURRED:
            obj_str += " BLURRED"
        elif self.__status & FrameStatus.MASKED:
            obj_str += " MASKED"

        obj_str += "), shape:"
        obj_str += str(self.image.shape)
        return obj_str

    @property
    def status(self) -> FrameStatus:
        """
        :return: Флаги статуса кадра
        """
        return self.__status

    @property
    def image(self) -> np.ndarray:
        """
        :return: Изображение карда
        """
        return self._image

    @property
    def fr_num(self) -> int:
        """
        :return: Номер кадра
        """
        return self._fr_num

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        :return: Размеры изображения
        """
        return self._image.shape[0:2]

    @property
    def m_mask(self) -> ndarray:
        """
        :return: Возвращает маску объекта на изображении.
        Если маска еще не задавалась, возвращает копию изображения.
        """
        if self._m_mask is None:
            return np.copy(self._image)
        return self._m_mask

    @property
    def mask(self) -> ndarray:
        """
        :return: Возвращает маску объекта на изображении.
        Если маска еще не задавалась, возвращает копию изображения.
        """
        if self._mask is None:
            return np.copy(self._image)
        return self._mask

    @property
    def object_present(self) -> bool:
        """
        :return: возвращает статус присутствия объекта на изображении
        """
        return self._object_present


    def resize(self, dims : Tuple[int, int]) -> 'Frame':
        """
        Приводит изображение и мастку объекта к заданному размеру, устанавливает флаг SHRINKED при уменьшении
        :param dims: Размеры нового изображения
        :return: self
        """
        if self._image.shape[0] > dims[0] and self._image.shape[1] > dims[1]:
            self.__status |= FrameStatus.SHRINKED

        self._image = bf.resize(self._image, dims[1], dims[0])
        if self.__status & FrameStatus.MASKED:
            self._mask = bf.resize(self._mask, dims[1], dims[0])

        if self.__status & FrameStatus.M_MASKED:
            self._m_mask = bf.resize(self._m_mask, dims[1], dims[0])

        return self

    def median_blur(self, blur_image : bool = False):
        """
        Производит медианную фильтрацию маски, устанавливает флаг BLURRED
        :param blur_image: При true фильтрует и само изображение
        :return: self
        """
        if blur_image:
            self._image = bf.blur(self._image, 7, 2)
            self.__status |= FrameStatus.BLURRED

        if self.__status & FrameStatus.MASKED:
            self._mask = bf.blur(self._mask, 7, 2)
            self.__status |= FrameStatus.BLURRED

        if self.__status & FrameStatus.M_MASKED:
            self._m_mask = bf.blur(self._m_mask, 7, 2)
            self.__status |= FrameStatus.BLURRED

        return self

    def add_mask(self, mask : ndarray) -> 'Frame':
        """
        Привязывает к кадру маску объекта, содержащегося на кадре, и устанавливает флаг MASKED
        :param mask: Маска объекта
        :return: self
        """
        self._mask = mask
        self.__status |= FrameStatus.MASKED
        return self

    def add_m_mask(self, mask : ndarray) -> 'Frame':
        """
        Привязывает к кадру движений объекта, содержащегося на кадре, и устанавливает флаг M_MASKED
        :param mask: Маска объекта
        :return: self
        """
        self._m_mask = mask
        self.__status |= FrameStatus.M_MASKED
        return self

    def count_new_pixels(self) -> int:
        """
        Производит подсчет белых пикселей маски
        :return: количество значимых пикселей; -1 если маска не задана
        """
        if self.__status & FrameStatus.M_MASKED:
            return np.sum(self.m_mask)
        else:
            return -1

    def check_object_presence(self, thresh : int) -> bool:
        """
        Определяет наличие объекта на кадре с помощью его маски, устанавливает флаг OBJECT_CHECKED
        :param thresh: количество значимых пикселей маски, при котором будет определяться присутствие объекта
        :return: True или FALSE -- статус наличия объекта на кадре
        """
        if self.__status & FrameStatus.MASKED:
            self.__status |= FrameStatus.OBJECT_CHECKED
            self._object_present = np.sum(self._mask) > thresh
            return self._object_present
        else:
            return False





