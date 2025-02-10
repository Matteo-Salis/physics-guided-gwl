from functools import partial

from utils.training_1d import *

def train_model(config):

    if config["loss"] == "data":
        
        print("Pure DL approach")
        return train_dl_model_1d
    
    elif config["loss"] == "data+pde":
        
        print("PIDL approach")
        
        num_cpoint_batch = config["num_cpoint_batch"]
        num_cpoint_instance = config["num_cpoint_instance"]
        coeff_loss_data = config["coeff_loss_data"]
        coeff_loss_pde = config["coeff_loss_pde"]
        
        return partial(train_dl_pde_model_1d, 
                        num_cpoint_batch = num_cpoint_batch,
                        num_cpoint_instance = num_cpoint_instance,
                        fdif_step = 0.0009,
                        coeff_loss_data = coeff_loss_data,
                        coeff_loss_pde = coeff_loss_pde)