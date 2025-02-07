from utils.training_1d import *

def train_model(config):

    if config["loss"] == "data":
        
        print("Pure DL approach")
        return train_dl_model_1d
    
    if config["loss"] == "data+pde":
        
        print("PIDL approach")
        coeff_loss_pde = config["coeff_loss_pde"]
        coeff_loss_data = config["coeff_loss_data"]
        
        forward_and_loss = partial(train_dl_pde_model_1d, 
                                            g = g,
                                            S_y = S_y,
                                            num_cpoints = num_cpoints,
                                            num_ctsteps = num_ctsteps,
                                            device = device,
                                            coeff_loss_data = coeff_loss_data,
                                            coeff_loss_pde = coeff_loss_pde)