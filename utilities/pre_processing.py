import os
from os.path import join
import glob
import shutil
import cv2
from utils import automkdir

'''
General purpose pre-processing functions up until training begins.
'''

# Copies images from one directory to another
def copy_to_loc(path, stop, name='new_folder', img_format='png'):
    folders = sorted(glob.glob(join(path, '*')))
    print("Folder list: {}".format(folders))
    for folder in folders:
        save_path = join(folder, name)
        automkdir(save_path)
        img_path = join(folder, 'truth_downsize_2')
        imgs = sorted(glob.glob(join(img_path, '*.{}'.format(img_format))))
        for i in range(stop):
            shutil.copy2(imgs[i], save_path)
        print("Successfully copied the first {} files from {} into {}"
              .format(stop, img_path, save_path))



'''This function accepts an input path, scale, name and image format as parameters
It finds all of the folders in the input path and for each folder, it will resize each of the
ground-truth images by the scale factor and save the files according to the name supplied'''
def resize_imgs_truth(path, scale=0.5, name='resize', img_format='png'):
    folders = sorted(glob.glob(join(path, '*')))
    # print("Folder list: {}".format(folders))
    for folder in folders:
        save_path = join(folder, name)
        automkdir(save_path)
        # print("Saved directory is: {}".format(save_path))
        img_path = join(folder, 'truth')
        # print("Input path: {}".format(img_path))
        imgs = sorted(glob.glob(join(img_path, '*.{}'.format(img_format))))
        # print(os.listdir(img_path))
        for img in range(len(imgs)):
            original = cv2.imread(imgs[img], cv2.IMREAD_UNCHANGED)
            modified = cv2.resize(original, (0, 0), fx=scale, fy=scale)
            filename = os.listdir(img_path)[img]
            cv2.imwrite(join(save_path, filename), modified)
            # print(filename)
        print('Successfully resized {} images by a scale of {} at {}'.format(len(imgs), scale, save_path))
    return print('Resize images concluded')


# Resize images by half using the resize method
def resize_img(img_path, scale=0.5):
    original = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # print("original shape: {}".format(original.shape))
    modified = cv2.resize(original, (0, 0), fx=scale, fy=scale)
    # print("modified shape: {}".format(modified.shape))
    # cv2.imshow('Original', original)
    # cv2.imshow('Modified', modified)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return modified


# Convert a video sequence into frames
def vid_to_frame(vid_name, format='png'):
    vid_path = "test\\additional\\{}".format(vid_name)
    save_path = 'test\\additional\\{}'.format(vid_name.rstrip('.mp4'))
    automkdir(save_path)
    vid_capture = cv2.VideoCapture(vid_path)
    success, image = vid_capture.read()
    count = 0
    while success:
        # cur_frame = "frame%d.png" % count
        cv2.imwrite(save_path + "\\frame%d.png" % count, image)
        success, image = vid_capture.read()
        print("Read frame {} {}".format(count, success))
        count += 1
