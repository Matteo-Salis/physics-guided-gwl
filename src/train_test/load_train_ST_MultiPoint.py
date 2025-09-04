from functools import partial
from train_test.training_ST_MultiPoint import *

def training_model(config):

    
    if config["dataset_type"] == "ST_MultiPoint":
        
            
        
        
        start_dates_plot_training = config["start_dates_plot_training"]
        n_pred_plot = config["n_pred_plot"]
        sensors_to_plot = config["sensors_to_plot"]
        map_t_step_to_plot = config["map_t_step_to_plot"]
        lat_lon_points = config["lat_lon_npoints"]
        plot_arch = config["plot_arch"]
        l2_alpha = config["l2_alpha"]
        plot_displacements = True if "_K" in config["model"] else False
        
        if config["physics"] is False:
            print("Training Approach: ST_MultiPoint Pure DL")
        
            return partial(pure_dl_trainer, 
                           start_dates_plot = start_dates_plot_training,
                           n_pred_plot = n_pred_plot,
                           sensors_to_plot = sensors_to_plot,
                           t_step_to_plot = map_t_step_to_plot,
                           lat_lon_points = lat_lon_points,
                           plot_arch = plot_arch,
                           l2_alpha = l2_alpha,
                           plot_displacements = plot_displacements)
            
        else:
            print("Training Approach: ST_MultiPoint Physics Informed")
            
            cpoints_start_epoch = config["cpoints_start_epoch"]
            print("cpoints_start_epoch: ", cpoints_start_epoch)
            
            coherence_alpha = config["coherence_alpha"]
            print("coherence_alpha: ", coherence_alpha)
            
            tstep_control_points = config["tstep_control_points"]
            print("tstep_control_points: ", tstep_control_points)
            
            reg_diffusion_eq = config["reg_diffusion_eq"]
            print("reg_diffusion_eq: ", reg_diffusion_eq)
            
            reg_delta_gw_l2 = config["reg_delta_gw_l2"]
            print("reg_delta_gw_l2: ", reg_delta_gw_l2)
            
            reg_delta_gw_l1 = config["reg_delta_gw_l1"]
            print("reg_delta_gw_l1: ", reg_delta_gw_l1)
            
            reg_K = config["reg_K"]
            print("reg_K: ", reg_K)
            
            reg_delta_s_l2 = config["reg_delta_s_l2"]
            print("reg_delta_s_l2: ", reg_delta_s_l2)
            
            reg_delta_s_l1 = config["reg_delta_s_l1"]
            print("reg_delta_s_l1: ", reg_delta_s_l1)
            
            reg_latlon_smoothness = config["reg_latlon_smoothness"]
            print("reg_latlon_smoothness: ", reg_latlon_smoothness)
            reg_temp_smoothness = config["reg_temp_smoothness"]
            print("reg_temp_smoothness: ", reg_temp_smoothness)
            
            return partial(physics_guided_trainer, start_dates_plot = start_dates_plot_training,
                           n_pred_plot = n_pred_plot,
                           sensors_to_plot = sensors_to_plot,
                           t_step_to_plot = map_t_step_to_plot,
                           lat_lon_points = lat_lon_points,
                           tstep_control_points = tstep_control_points,
                           plot_arch = plot_arch,
                           l2_alpha = l2_alpha,
                           cpoints_start_epoch = cpoints_start_epoch,
                           reg_diffusion_eq = reg_diffusion_eq,
                           coherence_alpha = coherence_alpha,
                           reg_delta_gw_l2 = reg_delta_gw_l2,
                           reg_delta_gw_l1 = reg_delta_gw_l1,
                           reg_K = reg_K,
                           reg_delta_s_l2 = reg_delta_s_l2,
                           reg_delta_s_l1 = reg_delta_s_l1,
                           reg_latlon_smoothness = reg_latlon_smoothness,
                           reg_temp_smoothness = reg_temp_smoothness,
                           plot_displacements = plot_displacements)
            
    else:
        print("No Dataset type found!!!")
        
if __name__ == "__main__":
    pass