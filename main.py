import os
from model.vespcn import VESPCN
# from model.ltdvsr import LTDVSR
# from model.mcresnet import MCRESNET
# from model.drvsr import DRVSR
# from model.frvsr import FRVSR
# from model.dufvsr import DUFVSR
from model.pfnl import PFNL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
if __name__=='__main__':
    model = PFNL()
    print('Model loaded!')
    #model=PFNL()
    model.train()
    print('Training finished')
    #Edit
    # model.testvideos('test/vid4')
    model.test_video('test/vid4/calendar')
    #model.testvideos('/dev/f/data/video/test2/udm10')
