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
udm10 = ['archpeople', 'archwall', 'auditorium', 'band', 'caffe', 'camera', 'clap', 'lake', 'photography', 'polyflow']
concat_list = vid4 + udm10
psnr_avgs = []
ssim_avgs = []
print(vid4 + udm10)
for i in range(len(concat_list)):
    if i < len(vid4):
        upper_dir = 'vid4'
    else:
        upper_dir = 'udm10'
    folder = concat_list[i]
    path_to_truth = F"./test/{upper_dir}/{folder}/truth"
    path_to_result = F"./test/{upper_dir}/{folder}/result_pfnl"
    psnr_vals = []
    ssim_vals = []
    for filename in os.listdir(path_to_truth):
        #print(filename)
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
    print("{}/{:<20} | Average PSNR: {:<10.2f} | Average SSIM: {:<10.4f}".format(upper_dir, folder, psnr_avg, ssim_avg))
    if i == (len(vid4)-1):
        print("{:25} | Average PSNR: {:<10.2f}| Average SSIM: {:<10.4f}".format(upper_dir, np.mean(psnr_avgs), np.mean(ssim_avgs)))
        psnr_avgs = []
        ssim_avgs = []
    elif i == (len(concat_list)-1):
        print("{:25} | Average PSNR: {:<10.2f}| Average SSIM: {:<10.4f}".format(upper_dir, np.mean(psnr_avgs), np.mean(ssim_avgs)))