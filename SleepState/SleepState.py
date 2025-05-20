import os
from calendar import day_abbr

import cv2
import numpy as np

from FrameClass.FrameClass import Frame
from SleepTranscription import SleepTranscription
from basic_functions import basic_functions as bf
from collections import Counter

class SleepTime:
    def __init__(self, hour, minute, second):
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)

        self.in_seconds = int(second + minute*60 + hour*3600)

    def __add__(self, other):
        hour, minute, second, in_seconds = 0,0,0,0
        if type(other) is SleepTime:
            hour = (other.hour + self.hour) % 24
            minute = (other.minute + self.minute) % 60
            second = (other.second + self.second) % 60

        elif type(other) is int or type(other) is float:
            in_seconds = (other + self.in_seconds) % (3600*24)

            hour = (in_seconds // 3600) % 24
            minute = (in_seconds // 60) % 60
            second = (in_seconds % 3600) % 60
        else:
            assert "Wrong datatype!"

        return SleepTime(hour, minute, second)

    def __sub__(self, other):
        hour, minute, second, in_seconds = 0,0,0,0
        if type(other) is SleepTime:
            hour = (self.hour - other.hour) % 24
            minute = (self.minute -  other.minute) % 60
            second = (self.second - other.second) % 60

        elif type(other) is int or type(other) is float:
            in_seconds = (self.in_seconds - other) % (3600*24)

            hour = (in_seconds // 3600) % 24
            minute = (in_seconds // 60) % 60
            second = (in_seconds % 3600) % 60
        else:
            assert "Wrong datatype!"

        return SleepTime(hour, minute, second)

    def __str__(self):
        res_str = ''
        res_str += str(self.hour)
        res_str += ':'
        res_str += str(self.minute)
        res_str += ':'
        res_str += str(self.second)

        return res_str

    def to_struct_time(self):
        pass

class SleepState:
    def __init__(self, pose_list, movement_intensity, is_present, starting_time, framerate = 1, epoch_len = 30, movement_threshhold = 0.01, wake_threshhold = 0.1):
        self.movement_intensity = movement_intensity
        self.is_present = is_present
        self.poses = pose_list

        self.stage_array = np.array((len(movement_intensity)))
        self.stage_dict = dict()
        self.epoch_len = int(epoch_len*framerate)

        self.starting_time = starting_time
        self.record_len_in_sec = int(len(movement_intensity)/framerate)

        self.calc_sleep_stage(threshhold_min = movement_threshhold, threshhold_max = wake_threshhold)
        self._get_stage_dict()

        self.sleep_duration_in_sec = self.record_len_in_sec - int(len(self.stage_array[self.stage_array == "WAKE"])/framerate)

    def _count_movments(self, threshhold):
        mov_dict = dict()

        prev_state = False
        for i in range(len(self.movement_intensity)):

            if self.movement_intensity[i] > threshhold:
                if not prev_state:
                    mov_dict[i] = True
                prev_state = True
            else:
                if prev_state:
                    mov_dict[i] = False
                prev_state = False
        return mov_dict

    def calc_sleep_stage(self, threshhold_min, threshhold_max):
        epoch_len = self.epoch_len
        # Заполняем массив данных значениями соответствующими небыстрому сну
        self.stage_array = np.array(["NREM" for i in range(len(self.movement_intensity))], dtype=str)

        # На промежутках
        for i in range(0, self.stage_array.shape[0], epoch_len):
            left_border = i-epoch_len//2 if (i-epoch_len//2) > 0 else i
            right_border = i+epoch_len//2 if (i+epoch_len//2) < len(self.stage_array) else len(self.stage_array)

            if not min(self.is_present[left_border:right_border]):
                self.stage_array[left_border:right_border] = "WAKE"
            elif np.sum(self.movement_intensity[left_border:right_border])/len(self.movement_intensity[left_border:right_border]) >= threshhold_max:
                self.stage_array[left_border:right_border] = "WAKE"
            elif threshhold_max > np.sum(self.movement_intensity[left_border:right_border])/len(self.movement_intensity[left_border:right_border]) > threshhold_min:
                self.stage_array[left_border:right_border] = "REM"

        return self.stage_array

    def _get_stage_dict(self):

        prev_state = ""
        self.stage_dict[str(self.starting_time-1)] = 'START'
        for i in range(self.stage_array.shape[0]):

            if self.stage_array[i] == "WAKE":
                if prev_state != "WAKE":
                    time = self.starting_time + (i/len(self.stage_array))*self.record_len_in_sec
                    self.stage_dict[str(time)] = "WAKE"
                prev_state = "WAKE"

            elif self.stage_array[i] == "REM":
                if prev_state != "REM":
                    time = self.starting_time + (i / len(self.stage_array)) * self.record_len_in_sec
                    self.stage_dict[str(time)] = "REM"
                prev_state = "REM"

            elif self.stage_array[i] == "NREM":
                if prev_state != "NREM":
                    time = self.starting_time + (i / len(self.stage_array)) * self.record_len_in_sec
                    self.stage_dict[str(time)] = "NREM"
                prev_state = "NREM"
        self.stage_dict[str(self.starting_time + self.record_len_in_sec)] = 'END'
        return self.stage_dict

    def _calc_TST(self):

        return self.sleep_duration_in_sec/3600

    def _calc_REM(self):
        return (np.sum(self.stage_array[:]=="REM") / np.sum(self.stage_array != "WAKE")) * self._calc_TST()

    def _calc_N_REM(self):
        vals = self.stage_dict.values()
        return Counter(vals)["REM"]

    def _calc_WAKE(self):
        return (np.sum(self.stage_array[:]=="WAKE") / len(self.stage_array)) * (self.record_len_in_sec / 3600)

    def _calc_N_WAKE(self):
        vals = self.stage_dict.values()
        return Counter(vals)["WAKE"]


    def _calc_NREM(self):
        return (np.sum(self.stage_array[:]=="NREM") / np.sum(self.stage_array != "WAKE")) * self._calc_TST()

    def _count_Pose(self):
        vals, counts = np.unique(self.poses[self.stage_array != "WAKE"], return_counts=True)

        return (np.max(counts) / np.sum(self.stage_array != "WAKE")) * self._calc_TST()

    def get_sleeping_score(self, Am=8.5, Ap=8.5, Aw = 2 , alpha = 1, beta = 0.01, gamma = 0.5):
        if np.min(np.asarray(self.stage_array) == "WAKE"):
            return (0,0,0)

        TST = self._calc_TST()
        TR = self._calc_REM()
        TW = self._calc_WAKE()
        TNR = self._calc_NREM()
        A = self._calc_N_WAKE()-1
        NR = self._calc_N_REM()
        P = self._count_Pose() # most_frequent_pose

        a_val = (Am/Ap)**2
        scores1 = (TST*a_val + TNR*a_val*0.5 + TR*a_val*0.5 - TW*a_val*0.5 - A/Aw)*Ap

        scores2 = (NR+A)/TST

        scores3 = (TNR/TST)*alpha + 100*beta + P*gamma

        print('TST = ', TST)
        print('TR = ', TR)
        print('TNR = ', TNR)
        print('TW = ', TW)
        print('NR = ', NR)
        print('A = ', A)

        return scores1, scores2, scores3





