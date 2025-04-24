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
record.read_cap_frames(3)

record.prosess_frames((120,80), 0.03, 5, 0.01)

record.save_frames("Test_class_f")
record.save_masks("Test_class_m")
record.save_motion_masks("Test_motions")
record.close_video()

list_objects = record.detect_object_frames(0.03)
list_motions = record.detect_motion(5, 0.01)

print(list_objects)
print(list_motions)