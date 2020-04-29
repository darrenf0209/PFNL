import tensorflow.compat.v1 as tf
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = 'data/train/HP/HP_000/truth/000.png'
img = tf.read_file(img_path)
img = tf.image.decode_png(img, channels=3)
print("image shape: {}".format(img.shape))
# Floor divsion used to ensure integer data type
offset_height = [0, img.shape[0]//2]
offset_width = [0, img.shape[1]//2]
target_height = img.shape[0]//2
print("target height: {}".format(target_height))
target_width = img.shape[1]//2
print("target width: {}".format(target_width))
for height in offset_height:
    for width in offset_width:
        print("Offset Height: {}, Offset Width: {}".format(height, width))
        img_bbox = tf.image.crop_to_bounding_box(img, height, width, target_height, target_width)
        out_stack = tf.stack(img_bbox, axis=0)

print(out_stack)

