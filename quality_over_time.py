import matplotlib.pyplot as plt
import numpy as np
import os
import json

# path = "test/vid4/alternative_3_20200528_20200825_feedback_debug.txt"
# path = "test/vid4/alternative_3_20200528_20200824_feedback_debug.txt"
# path = "test/udm10/alternative_2_20200527_best.txt"
# # path = "test/vid4/null_3_20200529_best.txt"
# f = open(path, 'r')
# frameData = json.load(f)
#
# for videoSequence in frameData:
#     psnr_arr = []
#     ssim_arr = []
#     for frames in frameData[videoSequence]['frame'][0]:
#         psnr = frameData[videoSequence]['frame'][0][frames][0]
#         ssim = frameData[videoSequence]['frame'][0][frames][1]
#         psnr_arr.append(psnr)
#         ssim_arr.append(ssim)
#     psnr_arr = np.array(psnr_arr)
#     ssim_arr = np.array(ssim_arr)
#     plt.subplot(121)
#     plt.plot(range(0, len(psnr_arr)), psnr_arr)
#     plt.title('{}: PSNR Across Frames'.format(videoSequence))
#     plt.xlabel('Frame')
#     plt.ylabel('PSNR')
#     plt.subplot(122)
#     plt.plot(range(0, len(ssim_arr)), ssim_arr)
#     plt.title('{}: SSIM Across Frames'.format(videoSequence))
#     plt.xlabel('Frame')
#     plt.ylabel('SSIM')
#     plt.show()
#
path1 = "test/vid4/alt_only_cur_downsize_20200916_delete.txt"
# path = "test/vid4/null_3_20200529_best.txt"
f1 = open(path1, 'r')
frameData1 = json.load(f1)
path2 = "test/vid4/alt_only_cur_downsize_20200916_info_recycle_20201002.txt"
# path = "test/vid4/null_3_20200529_best.txt"
f2 = open(path2, 'r')
frameData2 = json.load(f2)
print(frameData2['average of average'])

for (vid1, vid2) in zip(frameData1, frameData2):
    psnr_arr1 = []
    ssim_arr1 = []
    psnr_arr2 = []
    ssim_arr2 = []
    if vid1 == 'average of average' or vid2 == 'average of average':
        continue
    for (frames1, frames2) in zip(frameData1[vid1]['frame'][0],frameData2[vid2]['frame'][0]):
        psnr1 = frameData1[vid1]['frame'][0][frames1][0]
        ssim1 = frameData1[vid1]['frame'][0][frames1][1]
        psnr_arr1.append(psnr1)
        ssim_arr1.append(ssim1)

        psnr2 = frameData2[vid2]['frame'][0][frames2][0]
        ssim2 = frameData2[vid2]['frame'][0][frames2][1]
        psnr_arr2.append(psnr2)
        ssim_arr2.append(ssim2)

    psnr_arr1 = np.array(psnr_arr1)
    ssim_arr1 = np.array(ssim_arr1)
    psnr_arr2 = np.array(psnr_arr2)
    ssim_arr2 = np.array(ssim_arr2)
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.plot(range(0, len(psnr_arr1)), psnr_arr1, label='Bicubic')
    plt.plot(range(0, len(psnr_arr2)), psnr_arr2, label='DL Output Feedback')
    plt.legend()
    plt.title('{}: PSNR Across Frames'.format(vid1))
    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.subplot(122)
    plt.plot(range(0, len(ssim_arr1)), ssim_arr1, label='Bicubic')
    plt.plot(range(0, len(ssim_arr2)), ssim_arr2, label='DL Output Feedback')
    plt.title('{}: SSIM Across Frames'.format(vid1))
    plt.legend(loc='upper right')
    plt.xlabel('Frame')
    plt.ylabel('SSIM')
    plt.show()



