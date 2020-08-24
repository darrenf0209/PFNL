import os
from model.alternative_original_copy import PFNL_alternative_2
from model.alternative_original_copy import NAME as PROPOSED_LOSS
from model.control import PFNL_control
from model.control import NAME as CONTROL
from model.alternative import PFNL_alternative
from model.alternative import NAME as ALTERNATIVE
from model.null import PFNL_null
from model.null import NAME as NULL
import time
from utilities.pre_processing import resize_imgs_truth
from utilities.pre_processing import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Pre-processing raw data files before loading
    # resize_imgs_truth(path='test\\vid4\\', scale=0.25, name='truth_downsize_4')
    # resize_imgs_truth(path='test\\vid4\\', scale=0.5, name='DELETE_TRIAL')
    # slice_imgs_truth(path='test\\vid4\\', use='first', num_tiles=4, copy_original=True, name='DELETE_TRIAL')
    # copy_to_loc(path='test\\vid4\\', stop=2, name="DELETE_AGAIN")

    # Choose model
    # model = PFNL_control()
    # model = PFNL_null()
    model = PFNL_alternative()
    # model = PFNL_alternative_2()
    print('Model loaded!')


    # Training the model
    # model.train()
    print('Training finished')

    # Testing the model
    if isinstance(model, PFNL_control):
        NAME = CONTROL
    elif isinstance(model, PFNL_null):
        NAME = NULL
    elif isinstance(model, PFNL_alternative_2):
        NAME = PROPOSED_LOSS
    else:
        NAME = ALTERNATIVE

    model.testvideos('test\\vid4', name='{}_20200824_feedback_debug'.format(NAME))
    # model.testvideos('test\\udm10', name='{}_test_truth'.format(NAME))
    # model.testvideos('test\\udm10\\', name='{}'.format(NAME))

    print('Runtime finished')
