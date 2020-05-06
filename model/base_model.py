import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from os.path import join, exists
import glob
import random
import numpy as np
from PIL import Image
import scipy
import cv2
from utils import LoadImage, DownSample, DownSample_4D, BLUR, AVG_PSNR, depth_to_space_3D, DynFilter3D, LoadParams, \
    cv2_imread, cv2_imsave, get_num_params, automkdir
from modules.videosr_ops import imwarp_forward
import time
import os
from tqdm import trange, tqdm
from utilities.pre_processing import resize_img, tile_img
from slice_img import tf_resize_image, tf_tile_image
# NEW
import tensorflow.compat.v1 as tf


class VSR(object):
    def __init__(self):
        self.num_frames = 7
        self.scale = 4
        self.in_size = 32
        self.gt_size = self.in_size * self.scale
        self.eval_in_size = [128, 240]
        self.batch_size = 16
        self.eval_basz = 4
        self.learning_rate = 1e-3
        self.end_lr = 1e-4
        self.reload = True
        self.max_step = int(1.5e5 + 1)
        self.decay_step = 1.2e5
        self.train_dir = './data/filelist_train.txt'
        self.eval_dir = './data/filelist_val.txt'
        self.save_dir = './checkpoint'
        self.log_dir = './eval_log.txt'

    def hypothesis_pipeline(self):
        # Retrieve paths to all training files
        print("Reading training directory")
        pathlist = open(self.train_dir, 'rt').read().splitlines()
        print("There are {} video sequences".format(len(pathlist)))

        # Shuffle the training paths
        random.shuffle(pathlist)
        print("First folder in shuffled pathlist: {}".format(pathlist[0]))

        # Gather all images from shuffled pathlist
        gt_list_all = []
        for path in pathlist:
            gt_list = sorted(glob.glob(os.path.join(path, 'truth/*.png')))
            gt_list_all.append(gt_list)

        # Select a random video sequence
        rand_vid = random.randint(0, len(gt_list_all)-1)
        print("rand_vid index: {}".format(rand_vid))
        gt_vid = gt_list_all[rand_vid]
        print("gt_vid: {}".format(gt_vid))

        # Select a random index frame from selected video sequence
        rand_frame = random.randint(0, len(gt_vid) - self.num_frames)
        print(len(gt_vid) - self.num_frames)
        print("rand_frame index: {}".format(rand_frame))

        # Create a batch of length self.num_frames from the index frame
        gt_batch = gt_vid[rand_frame:rand_frame + self.num_frames]
        print("Batch_list: {}".format(gt_batch))
        tiled_imgs = tile_img(gt_batch[0])
        resized_img = resize_img(gt_batch[-1])
        resized_img = np.expand_dims(resized_img, axis=0)
        gt_batch = np.concatenate((tiled_imgs, resized_img), axis=0)

        # Debugging code to view the batch
        for i in range(len(gt_batch)):
            cv2.imshow("img_{}".format(i), gt_batch[i, :, :, :])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("new batch shape: {}".format(gt_batch.shape))


        # 7) View the exact parts of each image at this point

        # 8) Perform the remainder of data augmentation...

        # return batch...

    def frvsr_input_producer(self):
        def read_data():
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            # gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            gt = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])
            input, gt = prepprocessing(input, gt)
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.in_size, self.in_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale, [0]], axis=-1)
            size_gt = tf.concat([[self.num_frames], size[:2] * self.scale, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            gt.set_shape([self.num_frames, self.in_size * self.scale, self.in_size * self.scale, 3])
            return input, gt

        pathlist = open(self.train_dir, 'rt').read().splitlines()
        random.shuffle(pathlist)
        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth_downsize_2/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=self.batch_size * 2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3,
                                                capacity=self.batch_size * 2)
        return batch_in, batch_gt

    def double_input_producer(self):
        def read_data():
            idx0 = self.num_frames // 2
            data_seq = tf.random_crop(self.data_queue, [2, self.num_frames])
            input = tf.stack(
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][idx0]), channels=3)])
            input, gt = prepprocessing(input, gt)

            flip = tf.random_uniform((1, 3), minval=0.0, maxval=1.0, dtype=tf.float32, seed=None,
                                     name=None)  # if training gets worse, comment the data flip part out
            input = tf.where(flip[0][0] < 0.5, input, input[:, ::-1])
            input = tf.where(flip[0][1] < 0.5, input, input[:, :, ::-1])
            input = tf.where(flip[0][2] < 0.5, input, tf.transpose(input, perm=(0, 2, 1, 3)))
            gt = tf.where(flip[0][0] < 0.5, gt, gt[:, ::-1])
            gt = tf.where(flip[0][1] < 0.5, gt, gt[:, :, ::-1])
            gt = tf.where(flip[0][2] < 0.5, gt, tf.transpose(gt, perm=(0, 2, 1, 3)))
            print('Input producer shape: ', input.get_shape(), gt.get_shape())
            return input, gt

        def prepprocessing(input, gt=None):
            input = tf.cast(input, tf.float32) / 255.0
            gt = tf.cast(gt, tf.float32) / 255.0

            shape = tf.shape(input)[1:]
            size = tf.convert_to_tensor([self.in_size, self.in_size, 3], dtype=tf.int32, name="size")
            check = tf.Assert(tf.reduce_all(shape >= size), ["Need value.shape >= size, got ", shape, size])
            shape = control_flow_ops.with_dependencies([check], shape)

            limit = shape - size + 1
            offset = tf.random_uniform(tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit

            offset_in = tf.concat([[0], offset], axis=-1)
            size_in = tf.concat([[self.num_frames], size], axis=-1)
            input = tf.slice(input, offset_in, size_in)
            offset_gt = tf.concat([[0], offset[:2] * self.scale, [0]], axis=-1)
            size_gt = tf.concat([[1], size[:2] * self.scale, [3]], axis=-1)
            gt = tf.slice(gt, offset_gt, size_gt)

            input.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            gt.set_shape([1, self.in_size * self.scale, self.in_size * self.scale, 3])
            return input, gt

        pathlist = open(self.train_dir, 'rt').read().splitlines()
        random.shuffle(pathlist)
        with tf.variable_scope('input'):
            inList_all = []
            gtList_all = []
            for dataPath in pathlist:
                inList = sorted(glob.glob(os.path.join(dataPath, 'blur{}/*.png'.format(self.scale))))
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth_downsize_2/*.png')))
                inList_all.append(inList)
                gtList_all.append(gtList)
            inList_all = tf.convert_to_tensor(inList_all, dtype=tf.string)
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)

            self.data_queue = tf.train.slice_input_producer([inList_all, gtList_all], capacity=self.batch_size * 2)
            input, gt = read_data()
            batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3,
                                                capacity=self.batch_size * 2)
        return batch_in, batch_gt

    def single_input_producer(self):
        def read_data():
            # Randomly crops self.data_queue into a 1-D tensor with size between 1 and num_frames
            data_seq = tf.random_crop(self.data_queue, [1, self.num_frames])
            print("Data seq: {}".format(data_seq))
            # input = tf.stack([tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            # Ground truth is a stack of the converted uint8 input frames into one tensor
            gt = tf.stack(
                # Decode each PNG of the input data sequence into a uint8 tensor
                [tf.image.decode_png(tf.read_file(data_seq[0][i]), channels=3) for i in range(self.num_frames)])
            # gt = tf.stack([tf.image.decode_png(tf.read_file(data_seq[1][i]), channels=3) for i in range(self.num_frames)])

            print("sp: {}".format(tf.shape(gt)[1:]))
            tiled_img = tf_tile_image(gt, save=False)
            print("Tiled image shape: {}".format(tf.shape(tiled_img)))
            # last_image = images[1, :, :, :]
            # print(last_image)
            resized_image = tf_resize_image(gt, save=False)
            print("Resized image shape: {}".format(tf.shape(resized_image)))
            processed_images = tf.concat([tiled_img, resized_image], 0)
            # processed_images = tf.stack((tiled_img, resized_image), axis=1)
            print("Processed image batch: {}".format(tf.shape(processed_images)))

            input, gt = prepprocessing(processed_images)
            # input, gt = prepprocessing(gt)

            return input, gt

        def prepprocessing(gt=None):
            # number of frames, width, height and channels
            n, w, h, c = gt.shape
            print("num_frames: {}, width: {}, height: {}, channels: {}".format(n, w, h, c))
            # Retrieve the width, height and channels from the ground-truth
            sp = tf.shape(gt)[1:]
            print("sp: {}".format(sp))
            # Convert square to int32
            size = tf.convert_to_tensor([self.gt_size, self.gt_size, c], dtype=tf.int32)
            print("Size: {}".format(size))

            limit = sp - size + 1
            print("limit: {}".format(limit))
            # Offset contains random values from a uniform distribution after taking the modulo with limit
            offset = tf.random_uniform(sp.shape, dtype=size.dtype, maxval=size.dtype.max, seed=None) % limit
            print("offset: {}".format(offset))
            offset_gt = tf.concat([[0], offset[:2], [0]], axis=-1)
            print("offset_gt: {}".format(offset_gt))
            size_gt = tf.concat([[n], size], axis=-1)
            print("size_gt: {}".format(size_gt))

            gt = tf.slice(gt, offset_gt, size_gt)
            print("gt tf.slice: {}".format(gt))
            gt = tf.cast(gt, tf.float32) / 255.
            print("gt tf.cast: {}".format(gt))
            # Data augmentation scheme with random flip and rotations
            flip = tf.random_uniform((1, 3), minval=0.0, maxval=1.0, dtype=tf.float32, seed=None, name=None)
            gt = tf.where(flip[0][0] < 0.5, gt, gt[:, ::-1])
            print("gt flip[0][0]: {}".format(gt))
            gt = tf.where(flip[0][1] < 0.5, gt, gt[:, :, ::-1])
            print("gt flip[0][1]: {}".format(gt))
            gt = tf.where(flip[0][2] < 0.5, gt, tf.transpose(gt, perm=(0, 2, 1, 3)))
            print("gt flip[0][2]: {}".format(gt))
            inp = DownSample_4D(gt, BLUR, scale=self.scale)
            print("inp: {}".format(inp))
            gt = gt[n // 2:n // 2 + 1, :, :, :]
            print("gt: {}".format(gt))

            # inp.set_shape([self.num_frames, self.in_size, self.in_size, 3])
            inp.set_shape([self.num_frames + 3, self.in_size, self.in_size, 3])
            gt.set_shape([1, self.in_size * self.scale, self.in_size * self.scale, 3])
            print('Input producer shapes: LR: {}, HR: {}'.format(inp.get_shape(), gt.get_shape()))

            return inp, gt

        print("Reading training directory")
        pathlist = open(self.train_dir, 'rt').read().splitlines()
        # Shuffle the paths of training data to reduce variance, ensure model remains general and prevent overfitting
        print("Shuffling the training paths")
        random.shuffle(pathlist)
        # Context manager
        with tf.variable_scope('trainin'):
            gtList_all = []
            for dataPath in pathlist:
                # Retrieve the ground-truth images in the datapath and append to a single list
                gtList = sorted(glob.glob(os.path.join(dataPath, 'truth/*.png')))
                gtList_all.append(gtList)
            # Convert paths to ground-truth images to tensor strings
            gtList_all = tf.convert_to_tensor(gtList_all, dtype=tf.string)
            print("gtList_all: {}".format(gtList_all))
            print("There are {} video sequences, each with {} frames".format(gtList_all.shape[0], gtList_all.shape[1]))

            # Prepare the data queue by slicing the string tensors according to queue capacity
            self.data_queue = tf.train.slice_input_producer([gtList_all], capacity=self.batch_size * 2)
            # Pass the input
            input, gt = read_data()
            print("Shape of input: {}".format(input.shape))
            print("Shape of gt: {}".format(gt.shape))
            # batch_in, batch_gt = tf.train.batch([input, gt], batch_size=self.batch_size, num_threads=3,
            #                                     capacity=self.batch_size * 2)
        #     batch_in, batch_gt = tf.train.batch([input, [gt]], batch_size=self.batch_size, num_threads=3,
        #                                         capacity=5)
        # return batch_in, batch_gt
        return input, gt

    def forward(self, x):
        pass

    def build(self):
        pass

    def eval(self):
        pass

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess=tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

        return

    def save(self, sess, checkpoint_dir, step):
        model_name = "VSR"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading SR checkpoints...")
        model_name = "VSR"

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading checkpoints...{} Success".format(ckpt_name))
            return True
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test_video(self, path, name='result', reuse=False):
        pass

    def testvideos(self):
        pass


if __name__ == '__main__':
    model = VSR()
    model.train()
    # model.testvideos()
