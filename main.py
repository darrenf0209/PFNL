import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
########################################################
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
########################################################
# from model.vespcn import VESPCN
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
    #model.train()
    #print('Training finished')
    # model.eval()
    # model.testvideos('test\\udm10\\')
    model.testvideos('test\\vid4\\')
    # print('Finished')