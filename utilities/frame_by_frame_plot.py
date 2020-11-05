import matplotlib.pyplot as plt
import numpy as np
import json

'''
This script plots multiple frame-by-frame results for PSNR and SSIM for testing video sequences.
The raw results files are generated from eval_video.m using MATLAB.
'''

### Three sets of results ###

# Loading result text files into JSON format
path1 = "test/vid4/alt_only_cur_downsize_20200916_ff_0.txt"
f1 = open(path1, 'r')
frameData1 = json.load(f1)

path2 = "test/vid4/alt_only_cur_downsize_20200916_obj2_HR_10_20201008.txt"
f2 = open(path2, 'r')
frameData2 = json.load(f2)

path3 = "test/vid4/alt_only_cur_downsize_20200916_info_recycle_ff_0_20201002.txt"
f3 = open(path3, 'r')
frameData3 = json.load(f3)

# Iterate through each video sequence
for (vid1, vid2, vid3) in zip(frameData1, frameData2, frameData3):
    # Initialise result arrays
    psnr_arr1 = []
    ssim_arr1 = []
    psnr_arr2 = []
    ssim_arr2 = []
    psnr_arr3 = []
    ssim_arr3 = []
    # Do not plot the final average of average result from the test since it is not a video sequence
    if vid1 == 'average of average' or vid2 == 'average of average' or vid3 == 'average of average':
        continue
    #iterate through each frame
    for (frames1, frames2, frames3) in zip(frameData1[vid1]['frame'][0],frameData2[vid2]['frame'][0], frameData3[vid3]['frame'][0]):
        psnr1 = frameData1[vid1]['frame'][0][frames1][0]
        ssim1 = frameData1[vid1]['frame'][0][frames1][1]
        psnr_arr1.append(psnr1)
        ssim_arr1.append(ssim1)

        psnr2 = frameData2[vid2]['frame'][0][frames2][0]
        ssim2 = frameData2[vid2]['frame'][0][frames2][1]
        psnr_arr2.append(psnr2)
        ssim_arr2.append(ssim2)

        psnr3 = frameData3[vid3]['frame'][0][frames3][0]
        ssim3 = frameData3[vid3]['frame'][0][frames3][1]
        psnr_arr3.append(psnr3)
        ssim_arr3.append(ssim3)

    psnr_arr1 = np.array(psnr_arr1)
    ssim_arr1 = np.array(ssim_arr1)

    psnr_arr2 = np.array(psnr_arr2)
    ssim_arr2 = np.array(ssim_arr2)

    psnr_arr3 = np.array(psnr_arr3)
    ssim_arr3 = np.array(ssim_arr3)

    plt.figure(figsize=(14, 7))
    plt.subplot(121)

    plt.plot(range(0, len(psnr_arr1)), psnr_arr1, label='No Information Recycling')
    plt.plot(range(0, len(psnr_arr2)), psnr_arr2, label='Periodic HR')
    plt.plot(range(0, len(psnr_arr3)), psnr_arr3, label='Information Recycling')

    plt.legend(loc='lower right')
    plt.title('{}: PSNR Across Frames'.format(vid1))
    plt.xlabel('Frame')
    plt.ylabel('PSNR')

    plt.subplot(122)

    plt.plot(range(0, len(ssim_arr1)), ssim_arr1, label='No Information Recycling')
    plt.plot(range(0, len(ssim_arr2)), ssim_arr2, label='Periodic HR')
    plt.plot(range(0, len(ssim_arr3)), ssim_arr3, label='Information Recycling')

    plt.title('{}: SSIM Across Frames'.format(vid1))

    plt.legend(loc='lower right')
    plt.xlabel('Frame')
    plt.ylabel('SSIM')
    plt.show()




