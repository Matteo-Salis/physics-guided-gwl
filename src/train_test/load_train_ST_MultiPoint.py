from functools import partial
from train_test.training_ST_MultiPoint import *

def training_model(config):

    
    if config["dataset_type"] == "ST_MultiPoint":
        
            
        print("Training Approach: ST_MultiPoint Pure DL")
        
        start_dates_plot_training = config["start_dates_plot_training"]
        n_pred_plot = config["n_pred_plot"]
        sensors_to_plot = config["sensors_to_plot"]
        map_t_step_to_plot = config["map_t_step_to_plot"]
        lat_lon_points = config["lat_lon_npoints"]
        plot_arch = config["plot_arch"]
        l2_alpha = config["l2_alpha"]
        plot_displacements = True if "_K" in config["model"] else False
        
        if config["physics"] is False:
        
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
            
            coherence_alpha = config["coherence_alpha"]
            tstep_control_points = config["tstep_control_points"]
            
            return partial(physics_guided_trainer, start_dates_plot = start_dates_plot_training,
                           n_pred_plot = n_pred_plot,
                           sensors_to_plot = sensors_to_plot,
                           t_step_to_plot = map_t_step_to_plot,
                           lat_lon_points = lat_lon_points,
                           tstep_control_points = tstep_control_points,
                           plot_arch = plot_arch,
                           l2_alpha = l2_alpha,
                           coherence_alpha = coherence_alpha,
                           plot_displacements = plot_displacements)
            
    else:
        print("No Dataset type found!!!")
        
if __name__ == "__main__":
    pass