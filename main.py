import os
from model.alternative import PFNL_alternative
from model.alternative import NAME
import time
from utilities.pre_processing import resize_imgs_truth
from utilities.pre_processing import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    # resize_imgs_truth(path='test\\vid4\\', scale=0.25, name='truth_downsize_4')
    # resize_imgs_truth(path='test\\vid4\\', scale=0.5, name='DELETE_TRIAL')
    # slice_imgs_truth(path='test\\vid4\\', use='first', num_tiles=4, copy_original=True, name='DELETE_TRIAL')
    # copy_to_loc(path='test\\vid4\\', stop=2, name="DELETE_AGAIN")
    model = PFNL_alternative()
    # print('Model loaded!')
    model.train()
    # print('Training finished')
    # model.testvideos('test\\udm10\\', name='{}_best_{}'.format(NAME, int(time.time())))
    # model.testvideos('test\\vid4\\', name='LR_LR_DELETE_{}_{}'.format(NAME, int(time.time())))
    # print('Finished')