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
NAME = 'Weighted_Pixel_Loss_Mean_Loss_50_50_20200903'


# Class holding all of the PFNL functions
class PFNL_alternative_2(VSR):
    def __init__(self):
        # number of input frames to the network
        self.num_frames = 2
        # Super-resolution scale
        self.scale = 2
        # Takes <num_frames> 32x32 frames for the NLRB reduce computation cost
        self.in_size = 32
        self.gt_size = self.in_size * self.scale
        self.eval_in_size = [128, 240]
        # Batch sizes for training and evaluation
        self.batch_size = 1
        self.eval_basz = 1
        # initial learning rate and follow polynomial decay to 1e-4 after 120,000 iterations
        self.learning_rate = 0.2e-3
        self.end_lr = 1e-4
        self.reload = True
        self.max_step = int(2.5e5 + 1)
        self.decay_step = 1.2e5
        # Directories for training or validation images, saving checkpoints or logging information
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
            inp1 = NonLocalBlock(inp1, int(c) * (self.num_frames + 3) * 4, sub_sample=1, nltype=1,
                                 scope='nlblock_{}'.format(0))
            # print("NLRB Output inp1:{}".format(inp1))
            inp1 = tf.depth_to_space(inp1, 2)
            # print("Re-arrange depth into spatial data inp1:{}".format(inp1))
            # Concatenation
            inp0 += inp1
            # print("inp0+=inp1: {}".format(inp0))
            inp0 = tf.split(inp0, num_or_size_splits=(self.num_frames + 3), axis=-1)
            # print("inp0 split: {}".format(inp0))
            # 5x5 convolutional step, before entering the PFRB
            inp0 = [conv0(f) for f in inp0]
            # print("inp0 conv0: {}".format(inp0))
            # Last frame is the current frame (causal system)
            # bic = tf.image.resize_images(x[:, -1, :, :, :], [w * self.scale, h * self.scale], method=2)
            bic = tf.image.resize_images(x[:, self.num_frames // 2 + 1, :, :, :], [w * self.scale, h * self.scale],
                                         method=2)
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
        L_train = tf.placeholder(tf.float32,
                                 shape=[self.batch_size, (self.num_frames + 3), self.in_size, self.in_size, 3],
                                 name='L_train')
        L_eval = tf.placeholder(tf.float32, shape=[self.eval_basz, (self.num_frames + 3), in_h, in_w, 3], name='L_eval')
        # SR denotes the function of the super-resolution network
        SR_train = self.forward(L_train)
        print("L Train shape {}".format(tf.shape(L_train)))
        print("SR Train shape {}".format(tf.shape(SR_train)))
        SR_eval = self.forward(L_eval)
        # Charbonnier Loss Function (differentiable variant of L1 norm)
        pixel_loss = tf.reduce_mean(tf.sqrt((SR_train - H) ** 2 + 1e-6))
        print("Pixel loss shape: {}".format(pixel_loss.shape))

        ###
        loss1 = self.loss_func(H, SR_train, 0)
        loss2 = self.loss_func(H, SR_train, 1)
        loss3 = self.loss_func(H, SR_train, 2)
        print("Shape of each loss channel: {}".format(loss1.shape))
        proposed_loss = tf.reduce_mean(loss1 + loss2 + loss3)
        print("Shape of combined proposed loss: {}".format((loss1+loss2+loss3).shape))
        print("Shape of mean proposed loss: {}".format(proposed_loss.shape))

        # print("Total proposed loss: {}".format(tf.keras.backend.eval(proposed_loss[0, 0, 0:5])))
        weight = 0.5
        self.loss = weight * pixel_loss + (1 - weight) * proposed_loss
        eval_mse = tf.reduce_mean((SR_eval - H) ** 2, axis=[2, 3, 4])
        self.eval_mse = eval_mse


        ###

        # Evaluate mean squared error
        # eval_mse = tf.reduce_mean((SR_eval - H) ** 2, axis=[2, 3, 4])
        # self.loss, self.eval_mse = loss, eval_mse
        self.L, self.L_eval, self.H, self.SR = L_train, L_eval, H, SR_train

    def loss_func(self, H, output_img, color):

        # for color in range(H.shape[-1]):
        # Label patches along single color dimension
        label_img_c = H[0, :, :, :, color]
        label_img_c = tf.expand_dims(label_img_c, axis=-1)
        label_patches = tf.extract_image_patches(
            images=label_img_c,  # one color channel at a time
            ksizes=[1, 3, 3, 1],  # 3x3 kernel
            strides=[1, 1, 1, 1],  # Create a 3x3 patch for every pixel
            rates=[1, 1, 1, 1],  # go along every pixel
            padding='SAME')  # Patches outside the image are zero

        ## Uncomment for mean + random
        label_mean = tf.math.reduce_mean(label_patches, axis=-1)
        # random = tf.random.normal(H.shape[:4])
        random = tf.random.normal(tf.shape(label_mean))
        # print("Label_mean shape {}".format(label_mean.shape))
        # print("output img shape {}".format(output_img.shape))
        loss = tf.math.square(output_img[0, 0, :, :, color] - (label_mean - random))

        ## Random between min and max of patch
        # max_val = tf.reduce_max(label_patches, axis=-1)
        # min_val = tf.reduce_min(label_patches, axis=-1)
        # random = tf.random_uniform(tf.shape(min_val), minval=min_val, maxval=max_val)
        # loss = tf.math.square(output_img[0, 0, :, :, color] - random)
        return loss

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
        eval_gt = tf.placeholder(tf.float32, [None, (self.num_frames + 3), out_h, out_w, 3])
        eval_inp = DownSample(eval_gt, BLUR, scale=self.scale)
        # print("eval_inp: {}".format(eval_inp))

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
                # index = np.array([i for i in range(idx0 - (self.num_frames+3) + 1, idx0 + 1)])
                index = np.array([i for i in range(idx0 - self.num_frames + 1, idx0 + 1)])
                # print("Index: {}".format(index))
                index = np.clip(index, 0, max_frame - 1).tolist()
                # print("Index: {}".format(index))
                # gt = [cv2_imread(gtlist[i]) for i in index]
                # Tiling the previous image
                gt_prev = cv2_imread(gtlist[index[0]])
                height = gt_prev.shape[0]
                width = gt_prev.shape[1]
                top_left = gt_prev[0:height // 2, 0:width // 2]
                bottom_left = gt_prev[height // 2:height, 0:width // 2]
                top_right = gt_prev[0:height // 2, width // 2:width]
                bottom_right = gt_prev[height // 2:height, width // 2:width]
                # print("cropped image shape: {}".format(top_left.shape))

                # Resizing the current reference frame
                gt_cur = cv2_imread(gtlist[index[1]])
                gt_cur = cv2.resize(gt_cur, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                # cv2.imshow("TopLeft", top_left)
                # cv2.imshow("BottomLeft", bottom_left)
                # cv2.imshow("TopRight", top_right)
                # cv2.imshow("BottomRight", bottom_right)
                # cv2.imshow("Current", gt_cur)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                top_left_crop = top_left[height // 2 - out_h:height // 2, width // 2 - out_w:width // 2, :].astype(
                    np.float32) / 255.0
                bottom_left_crop = bottom_left[0:out_h, width // 2 - out_w:width // 2, :].astype(np.float32) / 255.0
                top_right_crop = top_right[height // 2 - out_h:height // 2, 0:out_w, :].astype(np.float32) / 255.0
                bottom_right_crop = bottom_right[0:out_h, 0:out_w, :].astype(np.float32) / 255.0

                # Original cropping done by authors
                # cropped_img_1 = cropped_img_1[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0
                # cropped_img_2 = cropped_img_2[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0
                # cropped_img_3 = cropped_img_3[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0
                # cropped_img_4 = cropped_img_4[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0

                gt_cur_crop = gt_cur[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0
                # print("gt_cur image shape: {}".format(gt_cur.shape))

                # cv2.imshow("TopLeftCrop", top_left_crop)
                # cv2.imshow("BottomLeftCrop", bottom_left_crop)
                # cv2.imshow("TopRightCrop", top_right_crop)
                # cv2.imshow("BottomRightCrop", bottom_right_crop)
                # cv2.imshow("CurrentCrop", gt_cur_crop)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                ''' changed'''
                batch_gt.append(
                    np.stack((top_left_crop, bottom_left_crop, gt_cur_crop, top_right_crop, bottom_right_crop), axis=0))

                # gt = [i[border:out_h + border, border:out_w + border, :].astype(np.float32) / 255.0 for i in gt]
                # batch_gt.append(np.stack(gt, axis=0))
                # print("batch_gt shape: {}".format(batch_gt))
                # print('length of gtlist: {}'.format(len(gtlist)))
                # print("length of gt: {}".format(len(gt)))

                if len(batch_gt) == self.eval_basz:
                    batch_gt = np.stack(batch_gt, 0)
                    # print("batch_gt: {}".format(batch_gt))
                    batch_lr = sess.run(eval_inp, feed_dict={eval_gt: batch_gt})
                    # ORIGINAL
                    mse_val = sess.run(self.eval_mse,
                                       feed_dict={self.L_eval: batch_lr,
                                                  self.H: batch_gt[:,
                                                          (self.num_frames + 3) // 2:(self.num_frames + 3) // 2 + 1]})

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
        LR, HR = self.alternative_pipeline()
        print("Training begin")
        print("From pipeline: LR: {}, HR: {}".format(LR, HR))
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

        # TF configures the session
        config = tf.ConfigProto()
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
        gs = sess.run(global_step)
        print("Current Global Step: {}".format(gs))
        losses = []
        for step in range(sess.run(global_step), self.max_step):
            if step > gs and step % 20 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Step:{}, loss:{}'.format(step, loss_v))
                losses.append(loss_v)
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
            # _, loss_v = sess.run([training_op, self.loss], feed_dict={self.L: LR, self.H: HR})

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
        # print("Input Path: {}".format(inp_path))
        imgs_arr = sorted(glob.glob(join(inp_path, '*.png')))
        # print("Image set: {}".format(imgs_arr))
        max_frame = len(imgs_arr)
        print("Number of frames: {}".format(max_frame))
        # still need this
        '''Only the first image needs to be read to retrieve 'truth' dimensions'''
        imgs = np.array([cv2_imread(i) for i in imgs_arr]) / 255.
        h, w, c = imgs[0].shape

        all_imgs = []
        frames_foregone = 5

        # Flattened matrix of relevant images (HR, LR) according to frames foregone
        for i in range(frames_foregone, len(imgs_arr) - frames_foregone):
            # Reading and downsizing current image
            cur_img = cv2_imread(imgs_arr[i])
            cur_img_downsize = cv2.resize(cur_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            cur_img_downsize = np.array(cur_img_downsize) / 255

            # Reading previous HR image and splitting into tiles
            prev_img = cv2_imread(imgs_arr[i - 1])
            prev_top_left = prev_img[0: h // 2, 0: w // 2]
            prev_bottom_left = prev_img[h // 2: h, 0: w // 2]
            prev_top_right = prev_img[0: h // 2, w // 2: w]
            prev_bottom_right = prev_img[h // 2: h, w // 2: w]

            prev_top_left = np.array(prev_top_left) / 255
            prev_bottom_left = np.array(prev_bottom_left) / 255
            prev_top_right = np.array(prev_top_right) / 255
            prev_bottom_right = np.array(prev_bottom_right) / 255

            # all_imgs.extend([prev_top_left, prev_bottom_left, cur_img_downsize, prev_top_right, prev_bottom_right])
            all_imgs.extend([prev_top_left, prev_bottom_left, cur_img_downsize, prev_top_right, prev_bottom_right])

        print("all_imgs shape: {}".format(len(all_imgs)))

        if part > max_frame:
            part = max_frame
        if max_frame % part == 0:
            num_once = max_frame // part
        else:
            num_once = max_frame // part + 1

        L_test = tf.placeholder(tf.float32,
                                shape=[num_once, (self.num_frames + 3), h // (2 * self.scale), w // (2 * self.scale),
                                       3],
                                name='L_test')

        SR_test = self.forward(L_test)
        if not reuse:
            # self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            # self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)

            self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)

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
        print("lrs shape: {}".format(lrs.shape))
        lr_list = []
        # Temporary tensor
        test = np.zeros((1, 5, h // 4, w // 4, c))
        print("test shape: {}".format(test.shape))

        # Stacking each batch of 5 input frames
        for i in range(part - 2 * frames_foregone):
            index = np.array([i for i in range(i + frames_foregone - self.num_frames + 1, i + frames_foregone + 1)])
            # print("index: {}".format(index))
            index = np.clip(index, 0, max_frame - 1).tolist()
            # print("index: {}".format(index))
            # lr_list.append(np.array([lrs[j] for j in index]))
            # print("relative frames: {}:{}".format(i * 5, i * 5 + 4))
            # lr_list.extend(np.array(lrs[i * 5: i * 5 + 5]))
            batch_lr = np.stack(np.array(lrs[i * 5: i * 5 + 5]))
            batch_lr = np.expand_dims(batch_lr, 0)
            # print("batch_lr shape: {}".format(batch_lr.shape))
            test = np.concatenate((test, batch_lr), axis=0)
            # print("test shape: {}".format(test.shape))

        # lr_list = np.array(lr_list)
        test = np.array(test[1:])
        # print("Shape of lr list: {}".format(lr_list.shape))
        # print("test shape: {}".format(test.shape))

        print('Save at {}'.format(save_path))
        print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))
        # h, w, c = lrs.shape[1:]

        # Performing VSR and saving images
        all_time = []
        for i in trange(part - 2 * frames_foregone):
            st_time = time.time()
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            # print('Num_once: {}'.format(num_once))
            # print("Lr_list index: {}:{}".format(i * num_once, (i + 1) * num_once))
            sr = self.sess.run(SR_test, feed_dict={L_test: test[i * num_once:(i + 1) * num_once]},
                               options=run_options)
            # sr = self.sess.run(SR_test, feed_dict={L_test: lr_list[i*5: i*5 + 5]},
            #                    options=run_options)
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

    def test_video_memory(self, path, name='memory_result', reuse=True, part=50):
        save_path = join(path, name)
        print("Save Path: {}".format(save_path))
        # Create the save path directory if it does not exist
        automkdir(save_path)
        inp_path = join(path, 'truth')
        # inp_path=join(path,'truth_downsize_2')
        # print("Input Path: {}".format(inp_path))
        imgs_arr = sorted(glob.glob(join(inp_path, '*.png')))
        # print("Image set: {}".format(imgs_arr))
        num_frames = len(imgs_arr)
        print("Number of frames: {}".format(num_frames))
        truth_img_dim = np.array(cv2_imread(imgs_arr[0])) / 255
        h, w, c = truth_img_dim.shape
        print("Original dimensions\nHeight: {}, Width: {}, Channels: {}".format(h, w, c))

        # Pre-processing the first two frames (after considering frames_foregone)
        first_batch = []
        frames_foregone = 5
        cur_img = cv2_imread(imgs_arr[frames_foregone])
        cur_img_downsize = cv2.resize(cur_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        cur_img_downsize = np.array(cur_img_downsize) / 255

        # Reading the previous HR image and splitting into tiles
        # Reading previous HR image and splitting into tiles
        prev_img = cv2_imread(imgs_arr[frames_foregone - 1])

        # Downsizing by an additional 2 (once-off)
        # prev_img = cv2.resize(prev_img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        # These are the 4 lots of X. These should not be passed into the network per se
        prev_top_left = prev_img[0: h // 2, 0: w // 2]
        prev_bottom_left = prev_img[h // 2: h, 0: w // 2]
        prev_top_right = prev_img[0: h // 2, w // 2: w]
        prev_bottom_right = prev_img[h // 2: h, w // 2: w]
        prev_top_left = np.array(prev_top_left) / 255
        prev_bottom_left = np.array(prev_bottom_left) / 255
        prev_top_right = np.array(prev_top_right) / 255
        prev_bottom_right = np.array(prev_bottom_right) / 255

        # 5 lots of 2X, which now needed to be passed into the network
        first_batch.extend([prev_top_left, prev_bottom_left, cur_img_downsize, prev_top_right, prev_bottom_right])
        print(prev_top_left.shape, cur_img_downsize.shape)

        if part > num_frames:
            part = num_frames
        if num_frames % part == 0:
            num_once = num_frames // part
        else:
            num_once = num_frames // part + 1

        # Dimensions being passed into the network!
        L_test = tf.placeholder(tf.float32,
                                shape=[num_once, (self.num_frames + 3), h // (2 * self.scale), w // (2 * self.scale),
                                       3],
                                name='L_test')

        SR_test = self.forward(L_test)
        if not reuse:
            self.img_hr = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='H_truth')
            self.img_lr = DownSample_4D(self.img_hr, BLUR, scale=self.scale)
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
        lrs = self.sess.run(self.img_lr, feed_dict={self.img_hr: first_batch}, options=run_options)
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
        sr = self.sess.run(SR_test, feed_dict={L_test: lrs}, options=run_options)
        all_time.append(time.time() - st_time)
        for j in range(sr.shape[0]):
            img = sr[j][0] * 255.
            img = np.clip(img, 0, 255)
            img = np.round(img, 0).astype(np.uint8)
            # Name of saved file. This should match the 'truth' format for easier analysis in future.
            cv2_imsave(join(save_path, 'Frame {:0>3}.png'.format(frames_foregone)), img)

        all_time = np.array(all_time)
        print('spent {} s in total and {} s in average'.format(np.sum(all_time), np.mean(all_time[1:])))
        # if num_frames > 0:
        #     all_time = np.array(all_time)
        #     cur_folder = path.lstrip('test\\udm10')
        #     cur_folder = cur_folder.lstrip('test\\vid4')
        #     time_dict = {
        #         "Folder": cur_folder,
        #         "Number of Frames": part - 2 * frames_foregone,
        #         "Total Time": np.sum(all_time),
        #         "Mean Time": np.mean(all_time[1:])
        #     }
        #     with open(self.test_dir, 'a+') as f:
        #         f.write(json.dumps(time_dict))
        #         f.write('\n')
        #     print('spent {} s in total and {} s in average'.format(np.sum(all_time), np.mean(all_time[1:])))

    '''
    This function accepts video frames of low quality and
    passes them through the network to perform 2xSR
    '''

    def test_video_lr(self, path, name='result', reuse=False, part=50):
        save_path = join(path, name)
        print("Save Path: {}".format(save_path))
        # Create the save path directory if it does not exist
        automkdir(save_path)
        # blurred_path = join(path, 'blur4')
        blurred_path = join(path, 'truth_downsize_4')
        truth_path = join(path, 'truth')
        # inp_path=join(path,'truth_downsize_4')
        # print("Input Path: {}".format(inp_path))
        # inp_path=join(path,'blur{}'.format(self.scale)) original
        imgs_blur = sorted(glob.glob(join(blurred_path, '*.png')))
        imgs_truth = sorted(glob.glob(join(truth_path, '*.png')))
        # print("Image set: {}".format(imgs))
        max_frame = len(imgs_blur)
        print("Number of frames: {}".format(max_frame))
        all_imgs_blur = np.array([cv2_imread(i) for i in imgs_blur]) / 255.
        all_imgs_truth = np.array([cv2_imread(i) for i in imgs_truth]) / 255.
        h, w, c = all_imgs_truth[0].shape

        all_imgs = []
        frames_foregone = 5

        for i in range(frames_foregone, len(imgs_truth) - frames_foregone):
            # Reading current LR frame
            cur_img = cv2.imread(imgs_blur[i])
            cur_img = np.array(cur_img) / 255

            # Reading previous HR image and splitting into tiles
            prev_img = cv2_imread(imgs_truth[i - 1])
            prev_top_left = prev_img[0: h // 2, 0: w // 2]
            prev_bottom_left = prev_img[h // 2: h, 0: w // 2]
            prev_top_right = prev_img[0: h // 2, w // 2: w]
            prev_bottom_right = prev_img[h // 2: h, w // 2: w]

            prev_top_left = cv2.resize(prev_top_left, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            prev_bottom_left = cv2.resize(prev_bottom_left, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            prev_top_right = cv2.resize(prev_top_right, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            prev_bottom_right = cv2.resize(prev_bottom_right, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

            prev_top_left = np.array(prev_top_left) / 255
            prev_bottom_left = np.array(prev_bottom_left) / 255
            prev_top_right = np.array(prev_top_right) / 255
            prev_bottom_right = np.array(prev_bottom_right) / 255
            # print(cur_img.shape, prev_top_left.shape)

            all_imgs.extend([prev_top_left, prev_bottom_left, cur_img, prev_top_right, prev_bottom_right])

        # lrs = np.array([cv2_imread(i) for i in imgs]) / 255.

        if part > max_frame:
            part = max_frame
        if max_frame % part == 0:
            num_once = max_frame // part
        else:
            num_once = max_frame // part + 1

        h, w, c = all_imgs[0].shape

        L_test = tf.placeholder(tf.float32, shape=[num_once, (self.num_frames + 3), h, w, 3], name='L_test')
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
        # max_frame = all_imgs[0].shape
        test = np.zeros((1, 5, h, w, c))
        print("test shape: {}".format(test.shape))
        for i in range(part - 2 * frames_foregone):
            index = np.array([i for i in range(i + frames_foregone - self.num_frames + 1, i + frames_foregone + 1)])
            # print("index: {}".format(index))
            index = np.clip(index, 0, max_frame - 1).tolist()
            # print("index: {}".format(index))
            # lr_list.append(np.array([lrs[j] for j in index]))
            # print("relative frames: {}:{}".format(i * 5, i * 5 + 4))
            # lr_list.extend(np.array(lrs[i * 5: i * 5 + 5]))
            batch_lr = np.stack(np.array(all_imgs[i * 5: i * 5 + 5]))
            batch_lr = np.expand_dims(batch_lr, 0)
            # print("batch_lr shape: {}".format(batch_lr.shape))
            test = np.concatenate((test, batch_lr), axis=0)
            # print("test shape: {}".format(test.shape))
        # for i in range(max_frame):
        #     index = np.array([i for i in range(i - (self.num_frames + 3) + 1, i + 1)])
        #     print("index: {}".format(index))
        #     index = np.clip(index, 0, max_frame - 1).tolist()
        #     print("index: {}".format(index))
        #     lr_list.append(np.array([lrs[j] for j in index]))
        # lr_list = np.array(lr_list)

        # print('Save at {}'.format(save_path))
        # print('{} Inputs With Shape {}'.format(lrs.shape[0], lrs.shape[1:]))
        # h, w, c = lrs.shape[1:]
        test = np.array(test[1:])
        all_time = []
        for i in trange(part - 2 * frames_foregone):
            st_time = time.time()
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            sr = self.sess.run(SR_test, feed_dict={L_test: test[i * num_once:(i + 1) * num_once]},
                               options=run_options)
            all_time.append(time.time() - st_time)
            for j in range(sr.shape[0]):
                img = sr[j][0] * 255.
                img = np.clip(img, 0, 255)
                img = np.round(img, 0).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2_imsave(join(save_path, '{:0>4}.png'.format(i * num_once + j)), img)

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
                self.test_video_lr(k, name=name, reuse=False, part=1000)
                # RNN
                # self.test_video_memory(k, name=name, reuse=False, part=1000)


if __name__ == '__main__':
    model = PFNL_alternative_2()
    model.train()
    model.testvideos()