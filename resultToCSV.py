import csv
import json

path1 = "test/udm10/bicubic.txt"
f1 = open(path1, 'r')
frameData1 = json.load(f1)
path2 = "test/udm10/alt_only_cur_downsize_20200916_info_recycle_ff_0_20201002.txt"
f2 = open(path2, 'r')
frameData2 = json.load(f2)

with open('report/obj2_recycling_vs_bicubic.csv', 'a', newline='') as file:
    fieldnames = ['Sequence',
                  'Bicubic Avg PSNR',
                  'Bicubic Avg SSIM',
                  'Information Recycling Avg PSNR',
                  'Information Recycling Avg SSIM'
                  ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for (vid1, vid2) in zip(frameData1, frameData2):
        print(vid1, vid2)
        if vid1 == 'average of average' or vid2 == 'average of average':
            writer.writerow({fieldnames[0]: vid1,
                             fieldnames[1]: frameData1['average of average'][0],
                             fieldnames[2]: frameData1['average of average'][1],
                             fieldnames[3]: frameData2['average of average'][0],
                             fieldnames[4]: frameData2['average of average'][1],
                             })

        else:
            writer.writerow({fieldnames[0]: vid1,
                             fieldnames[1]: frameData1[vid1]['average'][0],
                             fieldnames[2]: frameData1[vid1]['average'][1],
                             fieldnames[3]: frameData2[vid2]['average'][0],
                             fieldnames[4]: frameData2[vid2]['average'][1],
                             })
