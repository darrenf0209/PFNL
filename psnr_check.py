import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
########################################################
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
########################################################
import numpy
import math
import cv2
import tensorflow as tf

original = cv2.imread("test\\udm10\\archpeople\\truth\\0020.png")
contrast = cv2.imread("test\\udm10\\archpeople\\folder\\0020.png")
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(original,contrast)
print("PSNR_SO: {}".format(d))

# Compute PSNR over tf.uint8 Tensors.
psnr1 = tf.image.psnr(original, contrast, max_val=255)

# Compute PSNR over tf.float32 Tensors.
original = tf.image.convert_image_dtype(original, tf.float32)
contrast = tf.image.convert_image_dtype(contrast, tf.float32)
psnr2 = tf.image.psnr(original, contrast, max_val=1.0)
# psnr1 and psnr2 both have type tf.float32 and are almost equal.
print("TF PSNR1: {}, PSNR2: {}".format(psnr1, psnr2))
