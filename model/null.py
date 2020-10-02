import os
import tensorflow as tf
import sys
from tensorflow.python.ops import control_flow_ops
from os.path import join, exists
import glob
import random
import numpy as np
from PIL import Image
import scipy
import cv2
import json
import time
from tensorflow.python.layers.convolutional import Conv2D, conv2d
from utils import NonLocalBlock, DownSample, DownSample_4D, BLUR, cv2_imread, cv2_imsave, automkdir, end_lr_schedule
from tqdm import tqdm, trange
from model.base_model import VSR
# TensorFlow back-compatability
import tensorflow.compat.v1 as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
########################################################
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
########################################################

''' 
This is a modified version of PFNL by Darren Flaks.
'''
NAME = 'null_3_20200529'

# Class holding all of the PFNL functions
class PFNL_null(VSR):
    def __init__(self):
        # Initialize variables with respect to images, training, evaluating and directory locations
        # Takes <num_frames> 32x32 LR frames as input to compute calculation cost
        self.num_frames = 2
        self.scale = 2
        self.in_size = 32
        self.gt_size = self.in_size * self.scale
        self.eval_in_size = [128, 240]
        self.batch_size = 1
        self.eval_basz = 1
        # initial learning rate of 1e-3 and follow polynomial decay to 1e-4 after 120,000 iterations
        self.learning_rate = 0.2e-3
        self.end_lr = 1e-4
        self.reload = True
        # Number of iterations for training
        self.max_step = int(3e5 + 1)
        self.decay_step = 1.2e5
        self.train_dir = './data/filelist_train.txt'
        self.eval_dir = './data/filelist_val.txt'
        self.save_dir = './checkpoint/{}'.format(NAME)
        self.log_dir = './logs/{}.txt'.format(NAME)
        self.test_dir = './test/{}_test_time.txt'.format(NAME)

    def forward(self, x):
        # Filters: dimensionality of output space
        # Based on PFS-PS, set the convolutional layer filter as 64
        mf = 64
        # Kernel size: Height and width of the 2D convolution window
        dk = 3
        # Leaky ReLU activation function after each convolutional layer
        activate = tf.nn.leaky_relu
        # First design the network adopting proposed PFRBs, denoted PFS. The main body contains 20 PFRBs
        num_block = 20
        # n = number of layers
        # f1 = filters
        # w = width
        # h = height
        # c = channels (depth)
        n, f1, w, h, c = x.shape
        ki = tf.keras.initializers.glorot_normal(seed=None)
        # Stride length
        ds = 1
        with tf.variable_scope('nlvsr', reuse=tf.AUTO_REUSE) as scope:
            # First convolutional layer uses a 5x5 kernel for a big receptive field (the rest use a 3x3 kernel)
            conv0 = Conv2D(mf, 5, strides=ds, padding='same', activation=activate, kernel_initializer=ki, name='conv0')
            # Preparing the convolutional layers to be used in the PFRB
            conv1 = [Conv2D(mf, dk, strides=ds, padding='same', activation=activate, kernel_initializer=ki,
                            name='conv1_{}'.format(i)) for i in range(num_block)]
            # 1x1 conv is used to refine the deep feature map, to avoid too many parameters
            conv10 = [Conv2D(mf, 1, strides=ds, padding='same', activation=activate, kernel_initializer=ki,
                             name='conv10_{}'.format(i)) for i in range(num_block)]
            conv2 = [Conv2D(mf, dk, strides=ds, padding='same', activation=activate, kernel_initializer=ki,
                            name='conv2_{}'.format(i)) for i in range(num_block)]
            # Used for the 3x3 convolutional layers, as per the architecture
            convmerge1 = Conv2D(12, 3, strides=ds, padding='same', activation=activate, kernel_initializer=ki,
                                name='convmerge1')
            # convmerge2=Conv2D(12, 3, strides=ds, padding='same', activation=None, kernel_initializer=ki, name='convmerge2')

            # Creating I_0
            inp0 = [x[:, i, :, :, :] for i in range(f1)]
            # print("Creating I_0:{}".format(inp0))
            # Joining to the end
            inp0 = tf.concat(inp0, axis=-1)
            # print("Concatenating at end:{}".format(inp0))
            # Rearrange blocks of spatial data into depth; height and width dimensions are moved to depth
            inp1 = tf.space_to_depth(inp0, 2)
            # print("Re-arrange spatial data into depth inp1:{}".format(inp1))
            # Non Local Resblock
            inp1 = NonLocalBlock(inp1, int(c) * self.num_frames * 4, sub_sample=1, nltype=1,
                                 scope='nlblock_{}'.format(0))
            # print("NLRB Output inp1:{}".format(inp1))
            inp1 = tf.depth_to_space(inp1, 2)
            # print("Re-arrange depth into spatial data inp1:{}".format(inp1))
            # Concatenation
            inp0 += inp1
            # print("inp0+=inp1: {}".format(inp0))
            inp0 = tf.split(inp0, num_or_size_splits=self.num_frames, axis=-1)
            # print("inp0 split: {}".format(inp0))
            # 5x5 convolutional step, before entering the PFRB
            inp0 = [conv0(f) for f in inp0]
            # print("inp0 conv0: {}".format(inp0))
            # Last frame is the current frame (causal system)
            bic = tf.image.resize_images(x[:, -1, :, :, :], [w * self.scale, h * self.scale], method=2)
            # print("bic: {}".format(bic))

            # After the 5x5 conv layer, add in the num_blocks of PFRBs to make full extraction of both
            # inter-frame and temporal correlations among multiple LR frames
            for i in range(num_block):
                # I_1 obtained from the first 3x3 convolution. It denotes feature maps extracted
                inp1 = [conv1[i](f) for f in inp0]
                # print("I_1: {}".format(inp1))

                # All I_1 feature maps are concatenated, containing information from all input frames
                # I_1_merged has depth num_blocks x N, when taking num_blocks frames as input
                base = tf.concat(inp1, axis=-1)
                # print("I_1_merged: {}".format(base))

                # Undergo 1x1 convolution
                # Filter number set to distillate the deep feature map into a concise one, I_2
                base = conv10[i](base)
                # print("I_2: {}".format(base))

                # Feature maps contain: self-independent spatial information and fully maximised temporal information
                # I_3 denotes merged feature maps
                inp2 = [tf.concat([base, f], -1) for f in inp1]
                # print("I_3: {}".format(inp2))

                # Depth of feature maps is 2 x N and 3x3 conv layers are adopted to extract spatio-temporal information
                inp2 = [conv2[i](f) for f in inp2]
                # print("I_3_convolved: {}".format(inp2))

                # I_0 is added to represent residual learning - output and input are required to have the same size
                inp0 = [tf.add(inp0[j], inp2[j]) for j in range(f1)]
                # print("PFRB_output: {}".format(inp0))

            # Sub-pixel magnification: Merge and magnify information from PFRB channels to obtain a single HR image
            merge = tf.concat(inp0, axis=-1)
            # print('Sub-pixel magnification')
            # print('merge tf.concat: {}'.format(merge))
            merge = convmerge1(merge)
            # print('convmerge: {}'.format(merge))
            # Rearranges blocks of depth into spatial data; height and width taken out of the depth dimension
            # large1=tf.depth_to_space(merge,2)
            # Bicubically magnified to obtain HR estimate
            # out1=convmerge2(large1)
            out = tf.depth_to_space(merge, 2)
            # print('out: {}'.format(out))

        # HR estimate output
        return tf.stack([out + bic], axis=1, name='out')

    def build(self):
        in_h, in_w = self.eval_in_size
        # H is the corresponding HR centre frame
        H = tf.placeholder(tf.float32, shape=[None, 1, None, None, 3], name='H_truth')
        # I is L_train, representing the input LR frames
        L_train = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_frames, self.in_size, self.in_size, 3],
                                 name='L_train')
        L_eval = tf.placeholder(tf.float32, shape=[self.eval_basz, self.num_frames, in_h, in_w, 3], name='L_eval')
        # SR denotes the function of the super-resolution network
        SR_train = self.forward(L_train)
        SR_eval = self.forward(L_eval)
        # Charbonnier Loss Function (differentiable variant of L1 norm)
        # epsilon is empirically set to 10e-3 (error in code?)
        loss = tf.reduce_mean(tf.sqrt((SR_train - H) ** 2 + 1e-6))
        # Evaluate mean squared error
        eval_mse = tf.reduce_mean((SR_eval - H) ** 2, axis=[2, 3, 4])
        self.loss, self.eval_mse = loss, eval_mse
        self.L, self.L_eval, self.H, self.SR = L_train, L_eval, H, SR_train

    def eval(self):
        print('Evaluating ...')
        if not hasattr(self, 'sess'):
            sess = tf.Session()
            self.load(sess, self.save_dir)
        else:
            sess = self.sess
        print('Saved directory: {}'.format(self.save_dir))
        border = 8
        in_h, in_w = self.eval_in_size
        out_h = in_h * self.scale  # 256
        out_w = in_w * self.scale  # 480
        bd = border // self.scale

        eval_gt = tf.placeholder(tf.float32, [None, self.num_frames, out_h, out_w, 3])
        eval_inp = DownSample(eval_gt, BLUR, scale=self.scale)
        print("eval_inp: {}".format(eval_inp))

        filenames = open(self.eval_dir, 'rt').read().splitlines()  # sorted(glob.glob(join(self.eval_dir,'*')))
        # print("Filenames: {}".format(filenames))
        gt_list = [sorted(glob.glob(join(f, 'truth', '*.png'))) for f in filenames]
        center = 15
        batch_gt = []
        batch_cnt = 0
        mse_acc = None
        for gtlist in gt_list:
            max_frame = len(gtlist)
            # print("Max frame: {}".format(max_frame))
            for idx0 in range(center, max_frame, 32):
                index = np.array([i for i in range(idx0 - self.num_frames + 1, idx0 + 1)])
                # print("Index: {}".format(index))
                index = np.clip(index, 0, max_frame - 1).tolist()
                # print("Index: {}".format(index))
                gt = [cv2_imread(gtlist[i]) for i in index]
                # print("Original shape: {}".format(gt[0].shape))
                # Resizing images by half
                height = gt[0].shape[0]
                width = gt[0].shape[1]
                gt = [cv2.resize(gt[i], (width // 2, height // 2), interpolation=cv2.INTER_AREA) for i in range(len(gt))]
                # print("Modified shape: {}".format(gt[0].shape))
                gt = [i[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0 for i in gt]
                # print("Cropped shape: {}".format(gt[0].shape))
                batch_gt.append(np.stack(gt, axis=0))
                # print("batch_gt shape: {}".format(batch_gt))

                if len(batch_gt) == self.eval_basz:
                    batch_gt = np.stack(batch_gt, 0)
                    # print("batch_gt: {}".format(batch_gt))
                    batch_lr = sess.run(eval_inp, feed_dict={eval_gt: batch_gt})
                    mse_val = sess.run(self.eval_mse,
                                       feed_dict={self.L_eval: batch_lr,
                                                  self.H: batch_gt[:, self.num_frames // 2:self.num_frames // 2 + 1]})
                    # print("Batch LR {}".format(batch_lr))
                    # print("Batch gt {}".format(batch_gt))
                    # print("MSE Value: {}".format(mse_val))
                    if mse_acc is None:
                        mse_acc = mse_val
                    else:
                        mse_acc = np.concatenate([mse_acc, mse_val], axis=0)
                    batch_gt = []
                    print('\tEval batch {} - {} ...'.format(batch_cnt, batch_cnt + self.eval_basz))
                    batch_cnt += self.eval_basz
                    # print("MSE Acc: {}".format(mse_acc))

        psnr_acc = 10 * np.log10(1.0 / mse_acc)
        mse_avg = np.mean(mse_acc, axis=0)
        psnr_avg = np.mean(psnr_acc, axis=0)
        for i in range(mse_avg.shape[0]):
            tf.summary.scalar('val_mse{}'.format(i), tf.convert_to_tensor(mse_avg[i], dtype=tf.float32))
        print('Eval PSNR: {}, MSE: {}'.format(psnr_avg, mse_avg))
        mse_avg = (mse_avg * 1e6).astype(np.int64) / (1e6)
        psnr_avg = (psnr_avg * 1e6).astype(np.int64) / (1e6)
        return mse_avg.tolist(), psnr_avg.tolist()

    def train(self):
        print("Training begin")
        LR, HR = self.null_pipeline()
        print("From single_input_producer(): LR: {}, HR: {}".format(LR, HR))
        global_step = tf.Variable(initial_value=0, trainable=False)
        self.global_step = global_step
        print("Global step: {}".format(global_step))
        self.build()
        print("Build completed")
        lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.decay_step, end_learning_rate=self.end_lr,
                                       power=1.)
        print("learning rate lr: {}".format(lr))

        vars_all = tf.trainable_variables()
        # This print statement throws error: 'int' object has not attribute 'value'
        # print('Params num of all:',get_num_params(vars_all))

        training_op = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=vars_all, global_step=global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # TF configures the session
        config = tf.ConfigProto(gpu_options=gpu_options)
        # Attempt to allocate only as much GPU memory based on runtime allocations
        # Allocatee little memory, and as Sessions continues to run, more GPU memory is provided

        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess=tf.Session()
        self.sess = sess
        # Output tensors and metadata obtained when executing a session
        sess.run(tf.global_variables_initializer())
        # Save class adds the ability to save and restore variables to and from checkpoints
        # max_to_keep indicates the maximum number of recent checkpoint files to keep (default is 5)
        # keep_checkpoint_every_n_hours here means keep 1 checkpoint every hour of training
        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        if self.reload:
            self.load(sess, self.save_dir)
        # Mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph (deprecated)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Begin timing the training
        start_time = time.time()
        # print("start time: {}".format(start_time))
        gs = sess.run(global_step)
        print("Current global step: {}".format(gs))
        losses = []
        for step in range(sess.run(global_step), self.max_step):
            if step > gs and step % 20 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Step:{}, loss:{}'.format(step, loss_v))
                losses.append(loss_v)

                # LR, HR = self.null_pipeline()

            if (time.time() - start_time) > 5 and step % 500 == 0:
                print("Saving checkpoint")
                # if step > gs:
                #     self.save(sess, self.save_dir, step)
                self.save(sess, self.save_dir, step)
                training_time = time.time() - start_time
                print('Training cost time {}s.'.format(training_time))
                np_losses = np.array(losses)
                avg_loss = np.mean(np_losses)
                print("Average Loss from 500 Iterations: {}".format(avg_loss))
                mse_avg, psnr_avg = self.eval()

                cost_time = time.time() - start_time
                print('Training and evaluation cost {}s.'.format(cost_time))

                log_dict = {
                    "Date": time.strftime("%Y-%m-%d", time.localtime()),
                    "Time": time.strftime("%H:%M:%S", time.localtime()),
                    "Iteration": int(sess.run(self.global_step)),
                    "PSNR": float(psnr_avg[0]),
                    "MSE": float(mse_avg[0]),
                    "Loss": float(avg_loss),
                    "Training Time": training_time,
                    "Total Time": cost_time
                }

                with open(self.log_dir, 'a+') as f:
                    f.write(json.dumps(log_dict))
                    f.write('\n')
                print("Log complete")

                start_time = time.time()
                print("Timing restarted")
                self.end_lr = end_lr_schedule(step)
                # self.end_lr = end_lr_schedule(step) if end_lr_schedule(step) != "invalid" else self.end_lr
                print("Current end learning rate: {}".format(self.end_lr))


            lr1, hr = sess.run([LR, HR])
            _, loss_v = sess.run([training_op, self.loss], feed_dict={self.L: lr1, self.H: hr})

            if step > 500 and loss_v > 10:
                print('Model collapsed with loss={}'.format(loss_v))
                break

    '''
    This function accepts video frames of high quality and down-samples them by the scale factor
    It then passes the down-sampled imaged through the network to perform 2xSR
    '''

    def test_video_truth(self, path, name='result', reuse=True, part=50):
        save_path = join(path, name)
        print("Save Path: {}".format(save_path))
        # Create the save path directory if it does not exist
        automkdir(save_path)
        inp_path = join(path, 'truth')
        # inp_path=join(path,'truth_downsize_2')
        print("Input Path: {}".format(inp_path))
        imgs_arr = sorted(glob.glob(join(inp_path, '*.png')))
        print("Image set: {}".format(imgs_arr))
        max_frame = len(imgs_arr)
        print("Number of frames: {}".format(max_frame))

        imgs = np.array([cv2_imread(i) for i in imgs_arr]) / 255.

        if part > max_frame:
            part = max_frame
        if max_frame % part == 0:
            num_once = max_frame // part
        else:
            num_once = max_frame // part + 1

        h, w, c = imgs[0].shape
        all_imgs = []

        for i in range(len(imgs_arr)):
            img = cv2_imread(imgs_arr[i])
            img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            img = np.array(img) / 255
            all_imgs.append(img)

        L_test = tf.placeholder(tf.float32, shape=[num_once, self.num_frames, h // (2*self.scale), w // (2*self.scale), 3],
                                name='L_test')
        SR_test = self.forward(L_test)
        if not reuse:
            self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)
            print("img_lr: {}".format(self.img_lr))
            config = tf.ConfigProto()
            # Allow growth attempts to allocate only as much GPU memory based on runtime allocations
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.2
            sess = tf.Session(config=config)
            # sess=tf.Session()
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: all_imgs}, options=run_options)
        lr_list = []
        max_frame = lrs.shape[0]
        frames_foregone = 5
        for i in range(frames_foregone, max_frame - frames_foregone):
            index = np.array([i for i in range(i - self.num_frames + 1, i + 1)])
            # print("index: {}".format(index))
            index = np.clip(index, 0, max_frame - 1).tolist()
            print("index: {}".format(index))
            lr_list.append(np.array([lrs[j] for j in index]))
        lr_list = np.array(lr_list)

        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))
        h, w, c = lrs.shape[1:]

        all_time = []
        for i in trange(part - 2*frames_foregone):
            st_time = time.time()
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            sr = self.sess.run(SR_test, feed_dict={L_test: lr_list[i * num_once:(i + 1) * num_once]},
                               options=run_options)
            all_time.append(time.time() - st_time)
            for j in range(sr.shape[0]):
                img = sr[j][0] * 255.
                img = np.clip(img, 0, 255)
                img = np.round(img, 0).astype(np.uint8)
                # Name of saved file. This should match the 'truth' format for easier analysis in future.
                cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone + i * num_once + j + 1)), img)
        all_time = np.array(all_time)
        if max_frame > 0:
            all_time = np.array(all_time)
            cur_folder = path.lstrip('test\\udm10')
            cur_folder = cur_folder.lstrip('test\\vid4')
            time_dict = {
                "Folder": cur_folder,
                "Number of Frames": part - 2 * frames_foregone,
                "Total Time": np.sum(all_time),
                "Mean Time": np.mean(all_time[1:])
            }
            with open(self.test_dir, 'a+') as f:
                f.write(json.dumps(time_dict))
                f.write('\n')
            print('spent {} s in total and {} s in average'.format(np.sum(all_time), np.mean(all_time[1:])))

    # def test_video_memory(self, path, name='result', reuse=True, part=50):
    #     save_path = join(path, name)
    #     print("Save Path: {}".format(save_path))
    #     # Create the save path directory if it does not exist
    #     automkdir(save_path)
    #     inp_path = join(path, 'truth')
    #     print("Input Path: {}".format(inp_path))
    #     imgs_arr = sorted(glob.glob(join(inp_path, '*.png')))
    #     print("Image set: {}".format(imgs_arr))
    #     max_frame = len(imgs_arr)
    #     print("Number of frames: {}".format(max_frame))
    #
    #     if part > max_frame:
    #         part = max_frame
    #     if max_frame % part == 0:
    #         num_once = max_frame // part
    #     else:
    #         num_once = max_frame // part + 1
    #
    #     truth_dim = np.array(cv2_imread(imgs_arr[0]))/255
    #     h, w, c = truth_dim.shape
    #     all_imgs_halved = []
    #     for i in range(len(imgs_arr)):
    #         img = cv2_imread(imgs_arr[i])
    #         img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    #         img = np.array(img) / 255
    #         all_imgs_halved.append(img)
    #
    #     L_test = tf.placeholder(tf.float32,
    #                             shape=[num_once, self.num_frames, h // (2 * self.scale), w // (2 * self.scale), 3],
    #                             name='L_test')
    #     SR_test = self.forward(L_test)
    #     if not reuse:
    #         self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
    #         self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)
    #         print("img_lr: {}".format(self.img_lr))
    #         config = tf.ConfigProto()
    #         # Allow growth attempts to allocate only as much GPU memory based on runtime allocations
    #         config.gpu_options.allow_growth = True
    #         # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    #         sess = tf.Session(config=config)
    #         # sess=tf.Session()
    #         self.sess = sess
    #         sess.run(tf.global_variables_initializer())
    #         self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
    #         self.load(sess, self.save_dir)
    #     run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    #     lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: all_imgs_halved}, options=run_options)
    #     lr_list = []
    #     max_frame = lrs.shape[0]
    #     frames_foregone = 5
    #     for i in range(frames_foregone, max_frame - frames_foregone):
    #         index = np.array([i for i in range(i - self.num_frames + 1, i + 1)])
    #         # print("index: {}".format(index))
    #         index = np.clip(index, 0, max_frame - 1).tolist()
    #         print("index: {}".format(index))
    #         lr_list.append(np.array([lrs[j] for j in index]))
    #     lr_list = np.array(lr_list)
    #
    #     print('Save at {}'.format(save_path))
    #     print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))
    #     h, w, c = lrs.shape[1:]
    #
    #     all_time = []
    #     # SR on first image
    #     st_time = time.time()
    #     run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    #     sr = self.sess.run(SR_test, feed_dict={L_test: lr_list[frames_foregone:frames_foregone+1]},
    #                        options=run_options)
    #     all_time.append(time.time() - st_time)
    #     for j in range(sr.shape[0]):
    #         img = sr[j][0] * 255.
    #         img = np.clip(img, 0, 255)
    #         img = np.round(img, 0).astype(np.uint8)
    #         # Name of saved file. This should match the 'truth' format for easier analysis in future.
    #         cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone)), img)
    #
    #
    #     print(img.shape, all_imgs_halved[frames_foregone+1].shape)
    #     new_list = [img, all_imgs_halved[frames_foregone + 1]]
    #     cv2.imshow("HR estimate(prev)", img)
    #     cv2.imshow("Current LR", all_imgs_halved[frames_foregone+1])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #
    #     # lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: new_list}, options=run_options)
    #     # lrs = np.expand_dims(lrs, 0)
    #     # print(lrs.shape)
    #     #
    #     # sr = self.sess.run(SR_test, feed_dict={L_test: lrs[0:1]},
    #     #                    options=run_options)
    #     # all_time.append(time.time() - st_time)
    #     # for j in range(sr.shape[0]):
    #     #     img = sr[j][0] * 255.
    #     #     img = np.clip(img, 0, 255)
    #     #     img = np.round(img, 0).astype(np.uint8)
    #     #     # Name of saved file. This should match the 'truth' format for easier analysis in future.
    #     #     cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone+1)), img)
    #
    #     all_time = np.array(all_time)
    #     print('spent {} s in total and {} s in average'.format(np.sum(all_time), np.mean(all_time[1:])))
    def test_video_memory(self, path, name='memory_result', reuse=True, part=50):
        save_path = join(path, name)
        print("Save Path: {}".format(save_path))
        # Create the save path directory if it does not exist
        automkdir(save_path)
        # 4X
        # inp_path = join(path, 'truth')
        # 2X
        inp_path = join(path, 'truth_downsize_2')
        # print("Input Path: {}".format(inp_path))
        imgs_arr = sorted(glob.glob(join(inp_path, '*.png')))

        # Get the first SR image based on the truth dataset, before iteratively passing it through.

        # print("Image set: {}".format(imgs_arr))
        num_frames = len(imgs_arr)
        print("Number of frames: {}".format(num_frames))
        truth_img_dim = np.array(cv2_imread(imgs_arr[0])) / 255
        h, w, c = truth_img_dim.shape
        print("Truth - Height: {}, Width: {}, Channels: {}".format(h, w, c))

        # Pre-processing the first two frames, after considering frames_foregone
        first_batch = []
        frames_foregone = 1
        # Read 2X
        cur_img = np.array(cv2_imread(imgs_arr[frames_foregone])) / 255

        # Downsample to X

        # Reading the previous HR image and splitting into tiles
        prev_img = np.array(cv2_imread(imgs_arr[frames_foregone - 1])) / 255
        first_batch = np.stack((prev_img, cur_img), axis=0)




        if part > num_frames:
            part = num_frames
        if num_frames % part == 0:
            num_once = num_frames // part
        else:
            num_once = num_frames // part + 1

        # Dimensions being passed into the network!
        # L_test = tf.placeholder(tf.float32,
        #                         shape=[num_once, (self.num_frames + 3), h // (2 * self.scale), w // (2 * self.scale),
        #                                3],
        #                         name='L_test')
        L_test = tf.placeholder(tf.float32,
                                shape=[num_once, self.num_frames, h // self.scale, w // self.scale,
                                       3],
                                name='L_test')

        print("L_test shape: {}".format(L_test))
        SR_test = self.forward(L_test)
        print("SR_test shape: {}".format(SR_test))
        if not reuse:
            self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)
            print("self.img_lr shape: {}".format(self.img_lr.shape))
            config = tf.ConfigProto()
            # Allow growth attempts to allocate only as much GPU memory based on runtime allocations
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.2
            sess = tf.Session(config=config)
            # sess=tf.Session()
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # print("LR shape: {}".format(self.img_lr.shape))

        # img_hr_placeholder

        lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: first_batch}, options=run_options)

        # lrs = self.sess.run(first_batch_tensor, feed_dict={self.img_hr: first_batch}, options=run_options)
        print("lrs shape: {}".format(lrs.shape))
        lr_list = []
        lrs = np.expand_dims(lrs, 0)
        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))

        # Performing VSR and saving images
        all_time = []
        st_time = time.time()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        # print('Num_once: {}'.format(num_once))
        # print("Lr_list index: {}:{}".format(i * num_once, (i + 1) * num_once))
        # batch_trial = np.stack((prev_top_left, prev_bottom_left, lrs[0, 0], prev_top_right, prev_bottom_right))

        # sr = self.sess.run(SR_test, feed_dict={L_test: np.expand_dims(first_batch, 0)}, options=run_options)
        # sr = self.sess.run(SR_test, feed_dict={L_test: lrs}, options=run_options)

        sr = self.sess.run(SR_test, feed_dict={L_test: lrs}, options=run_options)
        print('sr output shape: {}'.format(sr.shape))
        # all_time.append(time.time() - st_time)
        for j in range(sr.shape[0]):
            img = sr[j][0] * 255.
            img = np.clip(img, 0, 255)
            img = np.round(img, 0).astype(np.uint8)
            # Name of saved file. This should match the 'truth' format for easier analysis in future.
            cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone)), img)

        # all_time = np.array(all_time)


        # Pass the output back into the network!

        # for i in trange(part - 2 * frames_foregone - 1):
        for i in range(2, part):
            new_batch = []

            # Tile the output image in preparation for feedback
            output_img = np.array(img) / 255


            # cur_img = np.array(cv2_imread(imgs_arr[frames_foregone+i])) / 255
            cur_img = np.array(cv2_imread(imgs_arr[i])) / 255


            new_batch.extend([output_img, cur_img])
            new_batch = np.array(new_batch)

            lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: new_batch}, options=run_options)
            sr = self.sess.run(SR_test, feed_dict={L_test: np.expand_dims(lrs, 0)}, options=run_options)

            # sr = self.sess.run(SR_test, feed_dict={L_test: np.expand_dims(new_batch, 0)}, options=run_options)
            # all_time.append(time.time() - st_time)
            for j in range(sr.shape[0]):
                img = sr[j][0] * 255.
                img = np.clip(img, 0, 255)
                img = np.round(img, 0).astype(np.uint8)
                # Name of saved file. This should match the 'truth' format for easier analysis in future.
                # cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone + i)), img)
                cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(i)), img)

    '''
    This function accepts video frames of low quality and
    passes them through the network to perform 2xSR
    '''

    def test_video_lr(self, path, name='result', reuse=False, part=50):
        save_path = join(path, name)
        print("Save Path: {}".format(save_path))
        # Create the save path directory if it does not exist
        automkdir(save_path)
        inp_path = join(path, 'blur4')
        # inp_path=join(path,'truth_downsize_4')
        print("Input Path: {}".format(inp_path))
        # inp_path=join(path,'blur{}'.format(self.scale)) original
        imgs = sorted(glob.glob(join(inp_path, '*.png')))
        print("Image set: {}".format(imgs))
        max_frame = len(imgs)
        print("Number of frames: {}".format(max_frame))
        lrs = np.array([cv2_imread(i) for i in imgs]) / 255.

        if part > max_frame:
            part = max_frame
        if max_frame % part == 0:
            num_once = max_frame // part
        else:
            num_once = max_frame // part + 1

        h, w, c = lrs[0].shape

        L_test = tf.placeholder(tf.float32, shape=[num_once, self.num_frames, h, w, 3], name='L_test')
        SR_test = self.forward(L_test)
        if not reuse:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # sess=tf.Session()
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)
            self.load(sess, self.save_dir)

        lr_list = []
        max_frame = lrs.shape[0]
        frames_foregone = 5
        for i in range(frames_foregone, max_frame - frames_foregone):
            index = np.array([i for i in range(i - self.num_frames + 1, i + 1)])
            print("index: {}".format(index))
            index = np.clip(index, 0, max_frame - 1).tolist()
            print("index: {}".format(index))
            lr_list.append(np.array([lrs[j] for j in index]))
        lr_list = np.array(lr_list)

        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))
        h, w, c = lrs.shape[1:]

        all_time = []
        for i in trange(part - 2*frames_foregone):
            st_time = time.time()
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            sr = self.sess.run(SR_test, feed_dict={L_test: lr_list[i * num_once:(i + 1) * num_once]},
                               options=run_options)
            all_time.append(time.time() - st_time)
            for j in range(sr.shape[0]):
                img = sr[j][0] * 255.
                img = np.clip(img, 0, 255)
                img = np.round(img, 0).astype(np.uint8)
                cv2_imsave(join(save_path, '{:0>4}.png'.format(frames_foregone + i * num_once + j)), img)

        all_time = np.array(all_time)
        if max_frame > 0:
            all_time = np.array(all_time)
            print('spent {} s in total and {} s in average'.format(np.sum(all_time), np.mean(all_time[1:])))

    # Default path written by authors
    def testvideos(self, path='/dev/f/data/video/test2/udm10', start=0, name='pfnl'):
        kind = sorted(glob.glob(join(path, '*')))
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
                print("Datapath: {}".format(k))
                # The datapath is not needed as the files are located at variable k
                # SR with HR as source
                # self.test_video_truth(k, name=name, reuse=False, part=1000)
                # SR with LR as source
                # self.test_video_lr(k, name=name, reuse=False, part=1000)
                # RNN
                self.test_video_memory(k, name=name, reuse=False, part=1000)


if __name__ == '__main__':
    model = PFNL_null()
    model.train()
    model.testvideos()
