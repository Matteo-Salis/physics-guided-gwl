from loss.losses_ST_MultiPoint import *
from functools import partial

def load_loss(config):
    
    if config["loss"] == "mse":
        print("Loss: MSE")
        return partial(loss_masked_mse)
    
    if config["loss"] == "mae":
        print("Loss: MAE")
        return partial(loss_masked_mae)
    
    if config["loss"] == "mape":
        print("Loss: MAPE")
        return partial(loss_masked_mape)
