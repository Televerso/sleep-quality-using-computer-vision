import numpy as np
import os
import cv2
from PIL import Image
import time
from FrameClass.FrameClass import *
from SleepState.SleepState import SleepState
from ViBe import vibe
from basic_functions import basic_functions as bf
from SleepTranscription.SleepTranscription import *
from datetime import datetime
from SleepState.SleepState import SleepTime

rootDir = "Sleeping vid 2"

record = SleepTranscription(rootDir)
# record.open_videofile("Test.mkv")
record.open_videofile("3.mkv")
gap = 30

print(record.get_total_cap_framecount())
record.set_cap_to_first_frame()
record.read_cap_frames((80,120), gap, 0)
print("Frames read")

record.prosess_frames_timed(0.01, 5, 0.02)

record.save_frames("Test_class_f")
record.save_masks("Test_class_m")
save_frames_list(record.motion_images_list, f"{rootDir}/Test_motions")

record.close_video()

list_objects = record.detect_object_frames(0.01)
list_motions = record.detect_motion(5)
list_poses = record.pose_list

list_keyframes = record.get_motion_frames(0.01)
list_key_masks = [frame.m_mask for frame in list_keyframes]
save_frames_list(list_key_masks, f"{rootDir}/Test_keyframes")

# dt = datetime.fromtimestamp(os.path.getctime(rootDir + "/Test.mkv"))
dt = datetime.fromtimestamp(os.path.getctime(rootDir + "/3.mkv"))
st = SleepTime(dt.hour, dt.minute, dt.second)

start = time.time()
sleep_quality = SleepState(record.pose_list, record.motion_detection_list, record.detect_object_frames(0.01), st, framerate=60/gap, epoch_len=210, movement_threshhold=0.0007 , wake_threshhold=0.006)
print(sleep_quality.get_sleeping_score())
end = time.time()
print(sleep_quality.stage_dict)
print("Sleep quality analysis time = ", (end-start)/len(sleep_quality.stage_array))