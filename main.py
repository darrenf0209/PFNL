import os
from model.generic_loss_functions import PFNL_generic_loss_functions
from model.generic_loss_functions import NAME as PROPOSED_LOSS
from model.control import PFNL_control
from model.control import NAME as CONTROL
from model.alternative import PFNL_alternative
from model.alternative import NAME as ALTERNATIVE
from model.null import PFNL_null
from model.null import NAME as NULL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    # Choose model
    model = PFNL_control()
    # model = PFNL_null()
    # model = PFNL_alternative()
    # model = PFNL_generic_loss_functions()
    print('Model loaded!')

    # Training the model
    model.train()
    print('Training finished')

    # Testing the specified model
    if isinstance(model, PFNL_control):
        NAME = CONTROL
    elif isinstance(model, PFNL_null):
        NAME = NULL
    elif isinstance(model, PFNL_generic_loss_functions):
        NAME = PROPOSED_LOSS
    else:
        NAME = ALTERNATIVE

    model.testvideos('test/vid4', name='{}'.format(NAME))
    model.testvideos('test/udm10', name='{}'.format(NAME))

    print('Runtime finished')
