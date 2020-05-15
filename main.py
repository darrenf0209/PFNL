import os
from model.control import PFNL_control
from model.control import NAME as CONTROL_NAME
from model.alternative import PFNL_alternative
from model.alternative import NAME as ALTERNATIVE_NAME
from model.null import PFNL_null
from model.null import NAME as NULL_NAME
import time
from utilities.pre_processing import resize_imgs_truth
from utilities.pre_processing import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    # Pre-processing raw data files before loading
    # resize_imgs_truth(path='test\\vid4\\', scale=0.25, name='truth_downsize_4')
    # resize_imgs_truth(path='test\\vid4\\', scale=0.5, name='DELETE_TRIAL')
    # slice_imgs_truth(path='test\\vid4\\', use='first', num_tiles=4, copy_original=True, name='DELETE_TRIAL')
    # copy_to_loc(path='test\\vid4\\', stop=2, name="DELETE_AGAIN")

    # Choosing the model
    # model = PFNL_control()
    # model = PFNL_alternative()
    model = PFNL_null()
    # print('Model loaded!')

    # Training the model
    model.train()
    # print('Training finished')

    # Testing the model
    # model.testvideos('test\\udm10\\', name='{}_best_{}'.format(NAME, int(time.time())))
    # model.testvideos('test\\vid4\\', name='LR_LR_DELETE_{}_{}'.format(NAME, int(time.time())))
    # print('Finished')
