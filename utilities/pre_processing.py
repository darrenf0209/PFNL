import os
from os.path import join
import glob
import shutil
import image_slicer
import cv2
from utils import automkdir


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


'''This function accepts an input path, number of tiles, name and image format as parameters
It finds all of the folders in the input path and for each folder, it will slice each of the
images into the number of tiles provided and save the files according to the name supplied.
The name of each saved image is the same as the original image name input with a suffix'''


def slice_imgs_truth(path, use, num_tiles, copy_original=False, name='truth_slice', img_format='png'):
    if use == 'nth':
        print('Slice every which image in folder?')
        nth_num = int(input())
    folders = sorted(glob.glob(join(path, '*')))
    print("Folder list: {}".format(folders))
    for folder in folders:
        save_path = join(folder, name)
        automkdir(save_path)
        print("Saved directory is: {}".format(save_path))
        img_path = join(folder, 'truth_downsize_2')
        # print("input path: {}".format(img_path))
        original_img_names = os.listdir(img_path)
        # print("Original image names: {}".format(original_img_names))
        imgs = sorted(glob.glob(join(img_path, '*.{}'.format(img_format))))
        # print("Set of images: {}".format(imgs))

        if use == 'first':
            img_prefix = "{}".format(original_img_names[0])
            img_prefix = img_prefix.rstrip('.{}'.format(img_format))
            tiles = image_slicer.slice(imgs[0], num_tiles, save=False)
            image_slicer.save_tiles(tiles, directory=save_path, prefix=img_prefix, format=img_format)
            if copy_original:
                shutil.copy2(imgs[1], save_path)

        elif use == 'nth':
            j = 0
            for img in range(len(imgs)):
                if j % nth_num == 0:
                    img_prefix = "{}".format(original_img_names[j])
                    img_prefix = img_prefix.rstrip('.{}'.format(img_format))
                    tiles = image_slicer.slice(imgs[j], num_tiles, save=False)
                    image_slicer.save_tiles(tiles, directory=save_path, prefix=img_prefix, format=img_format)
                else:
                    if copy_original:
                        shutil.copy2(imgs[j], save_path)
                j += 1

        elif use == 'all':
            print('Requested to slice all images in img path')
            for img in range(len(imgs)):
                img_prefix = "{}".format(original_img_names[img])
                img_prefix = img_prefix.rstrip('.{}'.format(img_format))
                tiles = image_slicer.slice(imgs[img], num_tiles, save=False)
                image_slicer.save_tiles(tiles, directory=save_path, prefix=img_prefix, format=img_format)
        else:
            raise ValueError("Argument 'use' must be either 'first' or 'nth' or 'all'")
        print("Successfully sliced {} into {} evenly dimensioned tiles at {}"
              .format(use, num_tiles, save_path))
    return print('Slicing images concluded')


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