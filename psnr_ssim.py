import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
########################################################
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
########################################################
import cv2
import tensorflow as tf

original = cv2.imread("test\\udm10\\polyflow\\truth\\0000.png")
contrast = cv2.imread("test\\udm10\\polyflow\\folder\\0000.png")

# Compute PSNR/SSIM over tf.uint8 Tensors.
original = tf.image.convert_image_dtype(original, tf.uint8)
contrast = tf.image.convert_image_dtype(contrast, tf.uint8)
psnr_uint8 = tf.image.psnr(original, contrast, max_val=255)
ssim_uint8 = tf.image.ssim(original, contrast, max_val=255, filter_size=11,
                           filter_sigma=1.5, k1=0.01, k2=0.03)

# Compute PSNR/SSIM over tf.float32 Tensors.
original = tf.image.convert_image_dtype(original, tf.float32)
contrast = tf.image.convert_image_dtype(contrast, tf.float32)
psnr_uint32 = tf.image.psnr(original, contrast, max_val=1.0)
ssim_uint32 = tf.image.ssim(original, contrast, max_val=1.0, filter_size=11,
                            filter_sigma=1.5, k1=0.01, k2=0.03)

# Both PSNR/SSIM calculations have type tf.float32 and are almost equal.
print("PSNR_uint8: {}, PSNR_uint32: {}".format(psnr_uint8, psnr_uint32))
print("SSIM_uint8: {}, SSIM_uint32: {}".format(ssim_uint32, ssim_uint32))