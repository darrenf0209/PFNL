import os
import glob
import cv2


'''This script applies bicubic up-sampling of the blurred and down-sampled by the scale provided to the specified 
video sequences. In this case, 2x up-sampling is applied to the testing video sequences. '''

scale = 2
paths = ("test/vid4", "test/udm10")
for path in paths:
    start = 0
    # Retrieve all folder paths
    kind = sorted(glob.glob(os.path.join(path, '*')))
    print("kind: {}".format(kind))
    kind = [k for k in kind if os.path.isdir(k)]
    reuse = False
    for k in kind:
        # index each folder
        idx = kind.index(k)
        print("idx: {}".format(idx))
        if idx >= start:
            if idx > start:
                reuse = True
            # Create a folder called "bicubic" at the location of original test sequences
            save_path = os.path.join(k, 'bicubic')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Input is the original down-sampled and blurred
            inp_path = os.path.join(k, 'blur4')
            imgs = sorted(glob.glob(os.path.join(inp_path, '*.png')))
            counter = 0
            for img in imgs:
                counter += 1
                cur_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                cv2.imshow("", cur_img)
                cv2.waitKey(100)

                # Dimensions for scaled up-sampling
                width = int(cur_img.shape[1] * scale)
                height = int(cur_img.shape[0] * scale)
                dim = (width, height)

                cur_img_upsize = cv2.resize(cur_img, dim, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("", cur_img_upsize)
                cv2.waitKey(100)
                # cv2.imwrite(os.path.join(save_path, 'Frame {:0>3}.png'.format(counter)), cur_img_upsize)

