import os
import glob
import cv2

scale = 2
paths = ("test/vid4", "test/udm10")
for path in paths:
    start = 0
    kind = sorted(glob.glob(os.path.join(path, '*')))
    print("kind: {}".format(kind))
    kind = [k for k in kind if os.path.isdir(k)]
    reuse = False
    for k in kind:
        idx = kind.index(k)
        print("idx: {}".format(idx))
        if idx >= start:
            if idx > start:
                reuse = True
            # datapath=join(path,k)
            # print("Datapath: {}".format(k))
            save_path = os.path.join(k, 'bicubic')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(save_path)
            inp_path = os.path.join(k, 'blur4')
            # print(inp_path)
            imgs = sorted(glob.glob(os.path.join(inp_path, '*.png')))
            print(imgs)
            counter = 0
            for img in imgs:
                counter += 1
                cur_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                # cv2.imshow("", cur_img)
                # cv2.waitKey(100)

                width = int(cur_img.shape[1] * scale)
                height = int(cur_img.shape[0] * scale)
                dim = (width, height)

                cur_img_upsize = cv2.resize(cur_img, dim, interpolation=cv2.INTER_CUBIC)
                # cv2.imshow("", cur_img_upsize)
                # cv2.waitKey(100)
                cv2.imwrite(os.path.join(save_path, 'Frame {:0>3}.png'.format(counter)), cur_img_upsize)


