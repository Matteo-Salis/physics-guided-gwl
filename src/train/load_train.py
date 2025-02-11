from functools import partial

from train.training_1d import *
from train.training_2d import *

def training_model(config):

    if  config["dataset_type"] == "1d":
    
        if config["loss"] == "data":
            
            print("Training Approach: 1d-Pure DL")
            
            return train_dl_model_1d
        
        elif config["loss"] == "data+pde":
            
            print("Training Approach: 1d-PIDL")
            
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
            
    elif config["dataset_type"] == "2d":
        
        print("Training Approach: 2d")
        
        c0_superres_loss = config["c0_superres_loss"]
        c1_masked_loss = config["c1_masked_loss"]
        c2_pde_darcy_loss = config["c2_pde_darcy_loss"]
        c3_positive_loss = config["c3_positive_loss"]

        h_timesteps = int(config["timesteps"]/2)
        timesteps = int(config["timesteps"])

        return partial(train_model_2d,
                        c0_superres_loss = c0_superres_loss,
                        c1_masked_loss = c1_masked_loss,
                        c2_pde_darcy_loss = c2_pde_darcy_loss,
                        c3_positive_loss = c3_positive_loss,
                        h_timesteps = h_timesteps,
                        timesteps = timesteps)