import os
from calendar import day_abbr

import cv2
import numpy as np

from FrameClass.FrameClass import Frame
from SleepTranscription import SleepTranscription
from basic_functions import basic_functions as bf
import time
from datetime import datetime
from collections import Counter

class sleep_time():
    def __init__(self, year, month, day, hour, minute, second):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second

    def to_struct_time(self):
        pass

class SleepState:
    def __init__(self, pose_list, is_moved, is_present, starting_time, framerate = 1, epoch_len = 30):
        self.is_moved = is_moved
        self.is_present = is_present
        self.poses = pose_list

        self.stage_array = np.array((len(is_moved)))
        self.stage_dict = dict()
        self.epoch_len = epoch_len

        self.starting_time = starting_time
        self.sleep_duration_in_sec = starting_time - len(is_moved)/framerate

    def _count_movments(self):
        mov_dict = dict()

        prev_state = False
        for i in range(len(self.is_moved)):

            if self.is_moved[i]:
                if not prev_state:
                    mov_dict[i] = True
                prev_state = True
            else:
                if prev_state:
                    mov_dict[i] = False
                prev_state = False
        return mov_dict

    def calc_sleep_stage(self):
        epoch_len = self.epoch_len
        # Заполняем массив данных значениями соответствующими небыстрому сну
        self.stage_array = np.array(["NREM" for i in range(len(self.is_moved))], dtype=str)

        # На промежутках
        for i in range(0, self.stage_array.shape[0], epoch_len):
            left_border = i-epoch_len//2 if (i-epoch_len//2) < 0 else i
            right_border = i+epoch_len//2 if (i+epoch_len//2) < 0 else i

            if not min(self.is_present[left_border:right_border]):
                self.stage_array[left_border:right_border] = "WAKE"

            elif max(self.is_moved[left_border:right_border]):
                self.stage_array[left_border:right_border] = "REM"

        return self.stage_array

    def _get_stage_dict(self):

        prev_state = ""
        for i in range(self.stage_array.shape[0]):

            if self.stage_array[i] == "WAKE":
                if prev_state != "WAKE":
                    self.stage_dict[i] = "WAKE"
                prev_state = "WAKE"

            elif self.stage_array[i] == "REM":
                if prev_state != "REM":
                    self.stage_dict[i] = "REM"
                prev_state = "REM"

            elif self.stage_array[i] == "NREM":
                if prev_state != "NREM":
                    self.stage_dict[i] = "NREM"
                prev_state = "NREM"

        return self.stage_dict

    def _calc_TST(self):
        return self.sleep_duration_in_sec

    def _calc_REM(self):
        return np.sum(self.stage_array[:]=="REM")

    def _calc_N_REM(self):
        vals = self.stage_dict.values()
        return Counter(vals)["REM"]

    def _calc_WAKE(self):
        return np.sum(self.stage_array[:]=="WAKE")

    def _calc_N_WAKE(self):
        vals = self.stage_dict.values()
        return Counter(vals)["WAKE"]


    def _calc_NREM(self):
        return np.sum(self.stage_array[:]=="NREM")

    def _count_Pose(self):
        counts, vals = np.unique(self.poses, return_counts=True)
        return np.max(counts)

    def get_sleeping_score(self, Am=8.5, Ap=8.5, Aw = 2 , alpha = 1, beta = 1, gamma = 0.5):
        TST = self._calc_TST()
        TR = self._calc_REM()
        TW = self._calc_WAKE()
        TNR = self._calc_NREM()
        A = self._calc_N_WAKE()
        NR = self._calc_N_REM()
        P = self._count_Pose() # most_frequent_pose

        a_val = (Am/Ap)**2
        scores1 = (TST*a_val + TNR*a_val*1.5 + TR*a_val*0.5 - TW*a_val*0.5 - A/Aw)*Ap

        scores2 = (NR+A)/TST

        scores3 = (TNR/TST)*alpha + 100*beta + P*gamma
        return scores1, scores2, scores3



