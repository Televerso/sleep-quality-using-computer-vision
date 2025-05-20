import os

import cv2

from FrameClass.FrameClass import Frame
from SleepTranscription.SleepTranscription import SleepTranscription
from PIL import Image


record = SleepTranscription("Sleeping vid 1")

frame_list = list()
for frame in os.listdir("Sleeping vid 1/Test_class_f")[:500]:
    image = cv2.imread(f"Sleeping vid 1/Test_class_f/{frame}")
    frame_list.append(Frame(image,0))

record.frame_list = frame_list
record.grayscale_frames()
record.do_ViBe_algorithm()
record.detect_object_frames()
record.blur_frames()



img_num = 0
for frame in record.frame_list:
    mask = frame.mask
    m_mask = frame.m_mask
    outfile_m = str("Sleeping vid 1/masks" + f'/{img_num}.png')
    outfile_mm = str("Sleeping vid 1/mmasks" + f'/{img_num}.png')
    img_num += 1
    image = Image.fromarray(mask)
    image.save(outfile_m)
    image = Image.fromarray(m_mask)
    image.save(outfile_mm)
