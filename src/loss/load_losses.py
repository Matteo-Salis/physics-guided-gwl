from loss.losses_2D import *

def load_loss(config):
    if config["loss"] == "mse":
        return loss_masked_mse

    if config["loss"] == "focal-mse":
        return loss_masked_focal_mse