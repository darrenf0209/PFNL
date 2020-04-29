import tensorflow.compat.v1 as tf
import time
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def tf_tile_image(img_4d, save=False):
    # (Num Frame, Height, Width, Channels)
    # if 4d image given, remove Num_Frame
    img_3d = tf.squeeze(img_4d)
    offset_height = [0, img_3d.shape[0] // 2]
    offset_width = [0, img_3d.shape[1] // 2]
    target_height = img_3d.shape[0] // 2
    print("target height: {}".format(target_height))
    target_width = img_3d.shape[1] // 2
    print("target width: {}".format(target_width))
    for height in offset_height:
        for width in offset_width:
            print("Offset Height: {}, Offset Width: {}".format(height, width))
            img_bbox = tf.image.crop_to_bounding_box(img_3d, height, width, target_height, target_width)
            if save:
                output_image = tf.image.encode_png(img_bbox)
                # Create a constant as filename
                file_name = tf.constant('./data/Output_image_{}_{}_{}.png'.format(height, width, time.time()))
                tf.write_file(file_name, output_image)
            out_stack = tf.stack(img_bbox, axis=0)
    return out_stack


img_path = ['data/train/HP/HP_000/truth/000.png', 'data/train/HP/HP_001/truth/000.png']
images = tf.stack(
    # Decode each PNG of the input data sequence into a uint8 tensor
    [tf.image.decode_png(tf.read_file(img_path[i]), channels=3) for i in range(len(img_path))])
print(images)
first_image = images[0, :, :, :]
print(first_image)
tf_tile_image(first_image)
last_image = images[1, :, :, :]
print(last_image)
tf_tile_image(last_image)
# first_image = tf.slice(images, [0, 0, 0, 0], [1, 720, 1272, 3])
# # print(tf.shape(first_image)[1:])
# # tile_image1 = tf_tile_image(first_image)
# last_image = tf.slice(images, [1, 0, 0, 0], [1, 720, 1272, 3])
# tile_image2 = tf_tile_image(last_image)
# print(last_image)


# img_path = 'data/train/HP/HP_000/truth/000.png'
# img = tf.read_file(img_path)
# img = tf.image.decode_png(img, channels=3)
# print(img)


# print("image shape: {}".format(img.shape))
# # Floor divsion used to ensure integer data type
# offset_height = [0, img.shape[0] // 2]
# offset_width = [0, img.shape[1] // 2]
# target_height = img.shape[0] // 2
# print("target height: {}".format(target_height))
# target_width = img.shape[1] // 2
# print("target width: {}".format(target_width))
# for height in offset_height:
#     for width in offset_width:
#         print("Offset Height: {}, Offset Width: {}".format(height, width))
#         img_bbox = tf.image.crop_to_bounding_box(img, height, width, target_height, target_width)
#         out_stack = tf.stack(img_bbox, axis=0)
#
# print(out_stack)
