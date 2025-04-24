import os

import cv2
import numpy as np

from FrameClass.FrameClass import Frame
from ViBe import vibe
from basic_functions import basic_functions as bf

class SleepTranscription:
    def __init__(self, rootDir : str):
        self.cap = None
        self.curr_cap_frame = 0
        self.frame_list = list()

        self.motion_masks_list = list()
        self.motion_detection_list = list()

        self.root = rootDir


    def open_videofile(self, filename):
        path_to_file = os.path.join(self.root, filename)
        self.cap = cv2.VideoCapture(path_to_file)

        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            return -1
        else:
            print("Video file opened successfully!")
            return 0

    def add_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_obj = Frame(frame, self.curr_cap_frame)
            self.curr_cap_frame += 1
            self.frame_list.append(frame_obj)
            return frame_obj
        else: return -1

    def set_cap_to_n_frame(self, n):
        self.cap.set(1, n)
        self.curr_cap_frame = n

    def set_cap_to_first_frame(self):
        self.cap.set(1, 0)
        self.curr_cap_frame = 0

    def get_total_cap_framecount(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def set_cap_to_last_frame(self):
        self.cap.set(1, self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        self.curr_cap_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1

    def read_cap_frames(self, gap):
        ret, frame = self.cap.read()
        i = 0
        while ret:
            if i%gap == 0:
                frame_obj = Frame(frame, self.curr_cap_frame)
                self.frame_list.append(frame_obj)

            self.curr_cap_frame += 1
            i += 1
            ret, frame = self.cap.read()

    def save_frames(self, path):
        img_num = 0
        for frame in self.frame_list:
            outfile = os.path.join(self.root, f'{path}/{img_num}.png')
            img_num += 1
            cv2.imwrite(outfile, frame.image)

    def save_masks(self, path):
        img_num = 0
        for frame in self.frame_list:
            outfile = os.path.join(self.root, f'{path}/{img_num}.png')
            img_num += 1
            cv2.imwrite(outfile, frame.mask)

    def save_motion_masks(self, path):
        img_num = 0
        for mask in self.motion_masks_list:
            outfile = os.path.join(self.root, f'{path}/{img_num}.png')
            img_num += 1
            cv2.imwrite(outfile, mask)

    def close_video(self):
        self.cap.release()
        self.cap = None


    def grayscale_frames(self):
        for frame in self.frame_list:
            frame.to_grayscale()
        return self.frame_list

    def resize_frames(self, dims):
        for frame in self.frame_list:
            frame.resize(dims)
        return self.frame_list

    def blur_frames(self):
        for frame in self.frame_list:
            frame.median_blur()
        return self.frame_list

    def do_ViBe_algorithm(self, params=(20,20,2,16), n_of_init_frame=0):
        N = params[0]
        R = params[1]
        _min = params[2]
        phai = params[3]

        samples = vibe.initial_background(self.frame_list[n_of_init_frame].image, N)
        for frame in self.frame_list:
            segMap, samples = vibe.vibe_detection(frame.image, samples, _min, N, R)
            frame.add_mask(segMap)
        return self.frame_list

    def detect_object_frames(self, threshold=0.05):
        new_pixel_thresh = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1] * threshold)

        for frame in self.frame_list:
            frame.check_motion(new_pixel_thresh)

        return [frame.motion for frame in self.frame_list]

    def detect_motion(self, n=5, threshold=0.02):
        new_pixel_thresh = int(255*self.frame_list[0].image_size[0] * self.frame_list[0].image_size[1] * threshold)

        for i in range(len(self.frame_list)):
            n_iter = n
            if i-n_iter < 0:
                n_iter = i
            frame = np.zeros_like(self.frame_list[i].mask)
            for j in range(1, n_iter):
                frame |= self.frame_list[i-j].mask ^ self.frame_list[i].mask
            self.motion_masks_list.append(frame)
            self.motion_detection_list.append(np.sum(frame) > new_pixel_thresh)
        return self.motion_detection_list

    def prosess_frames(self, dims, thresh_object, n, thresh_motion):
        self.grayscale_frames()
        self.resize_frames(dims)
        self.do_ViBe_algorithm()
        self.blur_frames()
        self.detect_object_frames()
        self.detect_motion()
        return self

