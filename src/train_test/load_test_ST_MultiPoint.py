from functools import partial
from train_test.test_ST_MultiPoint import *

def test_model(config):

    if config["dataset_type"] == "ST_MultiPoint":
        #if config["physics"] is False:
            
            print("Test Approach: ST_MultiPoint Pure DL")
            
            start_dates_plot_test = config["start_dates_plot_test"]
            n_pred_plot = config["n_pred_plot"]
            sensors_to_plot = config["sensors_to_plot"]
            map_t_step_to_plot = config["map_t_step_to_plot"]
            lat_lon_points = config["lat_lon_npoints"]
            plot_displacements = True if "_K" in config["model"] else False
            
            return partial(pure_dl_tester, 
                           start_dates_plot = start_dates_plot_test,
                           n_pred_plot = n_pred_plot,
                           sensors_to_plot = sensors_to_plot,
                           t_step_to_plot = map_t_step_to_plot,
                           lat_lon_points = lat_lon_points,
                           plot_displacements = plot_displacements)
            
    else:
        print("No Dataset type found!!!")
        
        
if __name__ == "__main__":
    pass