from loss.losses_2D import *
from functools import partial

def load_loss(config):
    if config["loss"] == "mse":
        return partial(loss_masked_mse, input = config["model"])
    
    if config["loss"] == "mae":
        return partial(loss_masked_mae, input = config["model"])
    
    if config["loss"] == "mape":
        return partial(loss_masked_mape, input = config["model"])
    
    if config["loss"] == "h2":
        return partial(loss_masked_h2, input = config["model"])
    
    if config["loss"] == "nse":
        return partial(loss_masked_nse, input = config["model"],
                       normalized = False)
    
    if config["loss"] == "nnse":
        return partial(loss_masked_nse, input = config["model"],
                       normalized = True)

    if config["loss"] == "focal-mse":
        return loss_masked_focal_mse
    
    if config["loss"] == "focal-mae":
        return loss_masked_focal_mae