import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
########################################################
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
########################################################
import cv2
import numpy as np
import glob
import pathlib
import tensorflow as tf

vid4 = ['calendar', 'city', 'foliage', 'walk']
psnr_avgs = []
ssim_avgs = []
# This generic approach works for udm10 as the file names are equivalent in 'truth' and 'result_pfnl
for folder in vid4:
    path_to_truth = F"./test/vid4/{folder}/truth"
    path_to_result = F"./test/vid4/{folder}/result_pfnl"
    # print(path_to_truth)
    psnr_vals = []
    ssim_vals = []
    for filename in os.listdir(path_to_truth):
        # print(filename)
        original = cv2.imread(os.path.join(path_to_truth, filename))
        contrast = cv2.imread(os.path.join(path_to_result, filename))
        original = tf.image.convert_image_dtype(original, tf.float32)
        contrast = tf.image.convert_image_dtype(contrast, tf.float32)
        psnr = tf.image.psnr(original, contrast, max_val=1.0)
        ssim = tf.image.ssim(original, contrast, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
    psnr_avg = np.mean(psnr_vals)
    psnr_avgs.append(psnr_avg)
    ssim_avg = np.mean(ssim_vals)
    ssim_avgs.append(ssim_avg)
    print("vid4/{:<10} | Average PSNR: {:<10.2f} | Average SSIM: {:<10.4f}".format(folder, psnr_avg, ssim_avg))
print("vid4 | Average of Averages PSNR: {:<10.2f}| Average of Averages SSIM: {:<10.4f}".format(np.mean(psnr_avgs), np.mean(ssim_avgs)))

udm10 = ['archpeople', 'archwall', 'auditorium', 'band', 'caffe', 'camera', 'clap', 'lake', 'photography', 'polyflow']
psnr_avgs = []
ssim_avgs = []
# This generic approach works for udm10 as the file names are equivalent in 'truth' and 'result_pfnl
for folder in udm10:
    path_to_truth = F"./test/udm10/{folder}/truth"
    path_to_result = F"./test/udm10/{folder}/result_pfnl"
    # print(path_to_truth)
    psnr_vals = []
    ssim_vals = []
    for filename in os.listdir(path_to_truth):
        # print(filename)
        original = cv2.imread(os.path.join(path_to_truth, filename))
        contrast = cv2.imread(os.path.join(path_to_result, filename))
        original = tf.image.convert_image_dtype(original, tf.float32)
        contrast = tf.image.convert_image_dtype(contrast, tf.float32)
        psnr = tf.image.psnr(original, contrast, max_val=1.0)
        ssim = tf.image.ssim(original, contrast, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
    psnr_avg = np.mean(psnr_vals)
    psnr_avgs.append(psnr_avg)
    ssim_avg = np.mean(ssim_vals)
    ssim_avgs.append(ssim_avg)
    print("udm10/{:<10} | Average PSNR: {:<10.2f} | Average SSIM: {:<10.4f}".format(folder, psnr_avg, ssim_avg))
print("udm10 | Average of Averages PSNR: {:<10.2f}| Average of Averages SSIM: {:<10.4f}".format(np.mean(psnr_avgs), np.mean(ssim_avgs)))