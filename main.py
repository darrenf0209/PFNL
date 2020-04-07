import os
from model.pfnl import PFNL
from model.pfnl import NAME
import time
from utils import resize_imgs_truth

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    # resize_imgs_truth(path='test\\udm10', scale=0.5, name='truth_downsize_2')
    model = PFNL()
    print('Model loaded!')
    # model.train()
    # print('Training finished')
    model.testvideos('test\\udm10\\', name='{}_{}'.format(NAME, int(time.time())))
    # model.testvideos('test\\vid4\\', name='{}_{}'.format(NAME, int(time.time())))
    # print('Finished')