from loss.losses_ST_MultiPoint import *
from functools import partial

def load_loss(config):
    
    if config["loss"] == "mse":
        return partial(loss_masked_mse)
    
    if config["loss"] == "mae":
        return partial(loss_masked_mae)
