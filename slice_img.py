import tensorflow.compat.v1 as tf
import time
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def tf_tile_image(img_4d, save=False):
    # (Num Frame, Height, Width, Channels)
    print(img_4d.shape)
    n, h, w, c = img_4d.shape
    offset_height = [0, h // 2]
    offset_width = [0, w // 2]
    target_height = h // 2
    # print("target height: {}".format(target_height))
    target_width = w // 2
    # print("target width: {}".format(target_width))
    temp_stack = tf.zeros([1, target_height, target_width, c], dtype=tf.uint8)
    for height in offset_height:
        for width in offset_width:
            # print("Offset Height: {}, Offset Width: {}".format(height, width))
            # Only do it to the first image
            img_bbox = tf.image.crop_to_bounding_box(img_4d[0, :, :, :], height, width, target_height, target_width)
            if save:
                # squeezed = tf.squeeze(img_bbox)
                output_image = tf.image.encode_png(img_bbox)
                # Create a constant as filename
                file_name = tf.constant('./data/Output_image_{}_{}.png'.format(height, width))
                tf.write_file(file_name, output_image)
            img_bbox = tf.expand_dims(img_bbox, 0)
            temp_stack = tf.concat([temp_stack, img_bbox], axis=0)
            print("temp stack shape during: {}".format(temp_stack.shape))
    print("temp stack after : {}".format(temp_stack.shape))
    output = temp_stack[1:, :, :, :]
    print("temp stack after slice : {}".format(output.shape))

    return output

def tf_resize_image(img_4d, scale=2, save=False):
    # (Batch, Height, Width, Channels)
    # if 4d image given, remove Num_Frame
    print(img_4d.shape)
    n, h, w, c = img_4d.shape
    new_height = h // scale
    new_width = w // scale
    resized_img = tf.image.resize(img_4d[1, :, :, :], (new_height, new_width))
    # print("New Height: {}, New Width: {}".format(resized_img.shape[0], resized_img.shape[1]))
    resized_img = tf.cast(resized_img, tf.uint8)
    print(" Before resized image shape: {}".format(resized_img.shape))
    resized_img = tf.expand_dims(resized_img, 0)
    print(" After resized image shape: {}".format(resized_img.shape))
    if save:
        output_image = tf.image.encode_png(resized_img[0, :, :, :])
        file_name = tf.constant('./data/Output_image_scale_{}.png'.format(scale))
        tf.write_file(file_name, output_image)
    return resized_img

img_path = ['data/train/HP/HP_000/truth/000.png', 'data/train/HP/HP_000/truth/001.png']
images = tf.stack(
    # Decode each PNG of the input data sequence into a uint8 tensor
    [tf.image.decode_png(tf.read_file(img_path[i]), channels=3) for i in range(len(img_path))])
print("Size of image set: {}".format(tf.shape(images)))
# first_image = images[0, :, :, :]
# print(first_image.shape)
tiled_img = tf_tile_image(images, save=False)
print("Tiled image shape: {}".format(tf.shape(tiled_img)))
# last_image = images[1, :, :, :]
# print(last_image)
resized_image = tf_resize_image(images, save=False)
print("Resized image shape: {}".format(tf.shape(resized_image)))
processed_images = tf.concat([tiled_img, resized_image], 0)
# processed_images = tf.stack((tiled_img, resized_image), axis=1)
print("Processed image batch: {}".format(tf.shape(processed_images)))
