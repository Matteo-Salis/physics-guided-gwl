from functools import partial
   
from train_test.test_1d import *
from train_test.test_2d import *

def test_model(config):

    if  config["dataset_type"] == "1d":
    
        if config["loss"] == "data":
            
            print("Test Approach: 1d-Pure DL")
            
            dates_list = config["test_plot_dates"]
            tsteps_list= config["test_plot_tstep_map"]
            
            return partial(test_dl_model_1d,
                           dates_list = dates_list,
                           tsteps_list = tsteps_list)
        
        elif config["loss"] == "data+pde":
            
            print("Test Approach: 1d-PIDL")
            
            dates_list = config["test_plot_dates"]
            tsteps_list= config["test_plot_tstep_map"]
            
            num_cpoint_batch = config["num_cpoint_batch"]
            num_cpoint_instance = config["num_cpoint_instance"]
            coeff_data_loss = config["coeff_data_loss"]
            coeff_pde_loss = config["coeff_pde_loss"]
            
            sampling_step = config["sampling_step"]
            fdif_step = config["fdif_step"]
            
            return partial(test_dl_pde_model_1d, 
                            num_cpoint_batch = num_cpoint_batch,
                            num_cpoint_instance = num_cpoint_instance,
                            sampling_step = sampling_step,
                            fdif_step = fdif_step,
                            coeff_data_loss = coeff_data_loss,
                            coeff_pde_loss = coeff_pde_loss,
                            dates_list = dates_list,
                            tsteps_list = tsteps_list)
            
    elif config["dataset_type"] == "2d":
        
        print("Test Approach: 2d")
        
        c0_superres_loss = config["coeff_superres_loss"]
        c1_masked_loss = config["coeff_data_loss"]
        c2_pde_darcy_loss = config["coeff_pde_loss"]
        c3_positive_loss = config["coeff_positive_loss"]

        h_timesteps = int(config["timesteps"]/2)
        timesteps = int(config["timesteps"])

        return partial(test_model_2d,
                        c0_superres_loss = c0_superres_loss,
                        c1_masked_loss = c1_masked_loss,
                        c2_pde_darcy_loss = c2_pde_darcy_loss,
                        c3_positive_loss = c3_positive_loss,
                        h_timesteps = h_timesteps,
                        timesteps = timesteps)
        
        
if __name__ == "__main__":
    pass