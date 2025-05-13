from functools import partial

from train_test.training_1d import *
from train_test.training_2d import *
from train_test.training_2D import *

def training_model(config):

    if  config["dataset_type"] == "1d":
    ######## 1D Approach ########
    
        if config["loss"] == "data":
            
            print("Training Approach: 1d-Pure DL")
            
            dates_list = config["train_plot_dates"]
            tsteps_list = config["train_plot_tstep_map"]
            
            teacher_training = config["teacher_training"]
            plot_arch = config["plot_arch"]
            
            if teacher_training is True:
                print("Teacher training")
            
            return partial(train_dl_model_1d, 
                           dates_list = dates_list,
                           tsteps_list = tsteps_list,
                           teacher_training = teacher_training,
                           plot_arch = plot_arch)
        
        elif config["loss"] == "data+pde":
            
            print("Training Approach: 1d-PIDL")
            
            dates_list = config["train_plot_dates"]
            tsteps_list= config["train_plot_tstep_map"]
            
            num_cpoint_batch = config["num_cpoint_batch"]
            num_cpoint_instance = config["num_cpoint_instance"]
            coeff_data_loss = config["coeff_data_loss"]
            coeff_pde_loss = config["coeff_pde_loss"]
            
            sampling_step = config["sampling_step"]
            fdif_step = config["fdif_step"]
            
            return partial(train_dl_pde_model_1d, 
                            num_cpoint_batch = num_cpoint_batch,
                            num_cpoint_instance = num_cpoint_instance,
                            sampling_step = sampling_step,
                            fdif_step = fdif_step,
                            coeff_data_loss = coeff_data_loss,
                            coeff_pde_loss = coeff_pde_loss,
                            dates_list = dates_list,
                            tsteps_list = tsteps_list)
            
    ######## 2d Approach ########  
    elif config["dataset_type"] == "2d":
    
        
        print("Training Approach: 2d")
        
        c0_superres_loss = config["coeff_superres_loss"]
        c1_masked_loss = config["coeff_data_loss"]
        c2_pde_darcy_loss = config["coeff_pde_loss"]
        c3_positive_loss = config["coeff_positive_loss"]

        h_timesteps = int(config["timesteps"]/2)
        timesteps = int(config["timesteps"])

        return partial(train_model_2d,
                        c0_superres_loss = c0_superres_loss,
                        c1_masked_loss = c1_masked_loss,
                        c2_pde_darcy_loss = c2_pde_darcy_loss,
                        c3_positive_loss = c3_positive_loss,
                        h_timesteps = h_timesteps,
                        timesteps = timesteps)
    
    ######## 2D Approach ########
    elif config["dataset_type"] == "2D_VideoCond":
        if config["physics"] is False:
            
            print("Training Approach: 2D-Pure DL")
            
            start_dates_plot_training = config["start_dates_plot_training"]
            twindow_plot = config["twindow_plot"]
            sensors_to_plot = config["sensors_to_plot"]
            timesteps_to_look = config["timesteps_to_look"]
            plot_arch = config["plot_arch"]
            
            return partial(train_dl_model, 
                           start_dates_plot = start_dates_plot_training,
                           twindow_plot = twindow_plot,
                           sensors_to_plot = sensors_to_plot,
                           timesteps_to_look = timesteps_to_look,
                           plot_arch = plot_arch)
        
if __name__ == "__main__":
    pass