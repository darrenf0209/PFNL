''' Created by Darren Flaks, last modified on 24/3/2020
    This file deals with the pre-processing of images before they are fed for training or testing '''

import os
import shutil
import image_slicer

vid4 = ['calendar', 'city', 'foliage', 'walk']
num_tiles = 16
# Variable for splitting every nth file into equally dimensioned tiles
nth = 10

for i in vid4:
    HR_path = F"./test/vid4/{i}/truth/"
    LR_path = F"./test/vid4/{i}/blur4/"
    save_path = F"./test/vid4/{i}/delete_this_{num_tiles}"

    HR_images = os.listdir(HR_path)
    # Image filenames across LR path and HR path may not be the same.
    LR_names = os.listdir(LR_path)

    # Create the directory to save files if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Counter of each file in HR image folder
    j = 0
    for HR_image in HR_images:
        # LR name is used to ensure the images in new directory are automatically sorted by name
        LR_name = F"{LR_names[j]}"
        # print(LR_name)
        # Split every nth file into equally dimensioned tiles
        if j % nth == 0:
            img_prefix = LR_name.rstrip('.png')
            # print(img_prefix)
            tiles = image_slicer.slice(HR_path + HR_image, num_tiles, save=False)
            image_slicer.save_tiles(tiles, directory=save_path, prefix=img_prefix, format='png')
            print("{} has been split into {} evenly dimensioned tiles".format(HR_image, num_tiles))
        else:
            # print(LR_path + LR_name)
            # Copy the LR file to the saved directory
            shutil.copy2(LR_path + LR_name, save_path)
        j += 1
