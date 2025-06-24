from loss.losses_2D import *
from functools import partial

def load_loss(config):
    if config["loss"] == "mse":
        return partial(loss_masked_mse, input = config["model"])
    
    if config["loss"] == "mae":
        return partial(loss_masked_mae, input = config["model"])

    if config["loss"] == "focal-mse":
        return loss_masked_focal_mse
    
    if config["loss"] == "focal-mae":
        return loss_masked_focal_mae