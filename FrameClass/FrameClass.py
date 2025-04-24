from enum import Flag, auto
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray

from ViBe import vibe
from basic_functions import basic_functions as bf


class FrameStatus(Flag):
    RGB = auto()
    GRAY = auto()
    SHRINKED = auto()
    MASKED = auto()
    BLURRED = auto()
    MOTION_CHECKED = auto()

class Frame:
    def __init__(self, frame : ndarray, fr_num : int):
        self._motion = False
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
        elif self.__status & FrameStatus.GRAY:
            obj_str += " GRAY"
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
        return self.__status

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def fr_num(self) -> int:
        return self._fr_num

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image.shape[0:2]

    @property
    def mask(self) -> ndarray:
        if self._mask is None:
            return np.copy(self._image)
        return self._mask

    @property
    def motion(self) -> bool:
        return self._motion

    def to_grayscale(self):
        self._image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.__status = FrameStatus.GRAY
        return self

    def resize(self, dims : Tuple[int, int]) -> 'Frame':
        if self._image.shape[0] > dims[0] and self._image.shape[1] > dims[1]:
            self.__status |= FrameStatus.SHRINKED

        self._image = bf.resize(self._image, dims[1], dims[0])
        if self.__status & FrameStatus.MASKED:
            self._mask = bf.resize(self._mask, dims[1], dims[0])
        return self

    def median_blur(self, blur_image = False):
        if blur_image:
            self._image = bf.blur(self._image, 7, 2)
            self.__status |= FrameStatus.BLURRED

        if self.__status & FrameStatus.MASKED:
            self._mask = bf.blur(self._mask, 7, 2)
            self.__status |= FrameStatus.BLURRED

        return self

    def add_mask(self, mask : ndarray) -> 'Frame':
        self._mask = mask
        self.__status |= FrameStatus.MASKED
        return self

    def count_new_pixels(self) -> int:
        if self.__status & FrameStatus.MASKED:
            return np.sum(self.mask)
        else:
            return -1

    def check_motion(self, thresh) -> bool:
        if self.__status & FrameStatus.MASKED:
            self.__status |= FrameStatus.MOTION_CHECKED
            self._motion = np.sum(self.mask) > thresh
            return self._motion
        else:
            return False





