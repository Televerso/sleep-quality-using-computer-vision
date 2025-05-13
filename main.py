import numpy as np
import os
import cv2
from PIL import Image

from FrameClass.FrameClass import *
from ViBe import vibe
from basic_functions import basic_functions as bf
from SleepTranscription.SleepTranscription import *

rootDir = "Moving vid"

record = SleepTranscription(rootDir)
record.open_videofile("video.mp4")

print(record.get_total_cap_framecount())
record.set_cap_to_last_frame()
record.add_next_frame()
record.set_cap_to_first_frame()
record.read_cap_frames(2)

record.prosess_frames((120,80), 0.01, 3, 0.02)

record.save_frames("Test_class_f")
record.save_masks("Test_class_m")
save_frames_list(record.motion_images_list, f"{rootDir}/Test_motions")

record.close_video()

list_objects = record.detect_object_frames(0.01)
list_motions = record.detect_motion(5, 0.03)
list_poses = record.pose_list
print(list_objects)
print(list_motions)
print(list_poses)

list_keyframes = record.get_motion_frames()
list_key_masks = [frame.mask for frame in list_keyframes]
save_frames_list(list_key_masks, f"{rootDir}/Test_keyframes")