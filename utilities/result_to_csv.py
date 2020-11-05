import csv
import json

'''
This script takes two sets of results and outputs a CSV summary. 
CSV files are easier to format for presentation purposes.
Result files are generated from eval_video.m in MATLAB.
'''

# Load two sets of results into JSON dataframes
path1 = "test/udm10/bicubic.txt"
f1 = open(path1, 'r')
frameData1 = json.load(f1)

path2 = "test/udm10/alt_only_cur_downsize_20200916_info_recycle_ff_0_20201002.txt"
f2 = open(path2, 'r')
frameData2 = json.load(f2)

# Create the CSV file with appropriate headers
with open('../report/obj2_recycling_vs_bicubic.csv', 'a', newline='') as file:
    fieldnames = ['Sequence',
                  'Bicubic Avg PSNR',
                  'Bicubic Avg SSIM',
                  'Information Recycling Avg PSNR',
                  'Information Recycling Avg SSIM'
                  ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Record results for each video sequence
    for (vid1, vid2) in zip(frameData1, frameData2):
        print(vid1, vid2)
        # Recording the overall average of the specified testing sequence
        if vid1 == 'average of average' or vid2 == 'average of average':
            writer.writerow({fieldnames[0]: vid1,
                             fieldnames[1]: frameData1['average of average'][0],
                             fieldnames[2]: frameData1['average of average'][1],
                             fieldnames[3]: frameData2['average of average'][0],
                             fieldnames[4]: frameData2['average of average'][1],
                             })
        # Average of a specific testing sequence
        else:
            writer.writerow({fieldnames[0]: vid1,
                             fieldnames[1]: frameData1[vid1]['average'][0],
                             fieldnames[2]: frameData1[vid1]['average'][1],
                             fieldnames[3]: frameData2[vid2]['average'][0],
                             fieldnames[4]: frameData2[vid2]['average'][1],
                             })
