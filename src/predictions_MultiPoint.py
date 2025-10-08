from operator import itemgetter
from tqdm import tqdm
import time
from datetime import datetime
import json
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray
import fiona

#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#from rasterio.enums import Resampling

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
from torch.autograd import Variable
from torchview import draw_graph

from utils.plot import *
import seaborn as sns
import argparse

import sys
import os

### Import Module ###

from dataloaders import dataset_ST_MultiPoint
from models import models_ST_MultiPoint
from models import load_model_ST_MultiPoint
from utils import plot_ST_MultiPoint
from utils import metrics

from utils.test_predictions import load_model
from utils.test_predictions import compute_prediction_with_displacement
from utils.test_predictions import compute_prediction

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--output', default=None, type=str, help='wandb.json file to track runs')
    args = parser.parse_args()
    return args

## Load dataset
def main(config):
    device = (
    config["device"]
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print("Device: ", device)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_names_dir = "_".join(config["model_name"])
    save_dir = '{}/{}_{}'.format(config["save_dir"],model_names_dir,timestamp)
    
    # Create Saving Directories
    os.makedirs(save_dir)
    ts_saving_path = save_dir+"/time_series"
    map_saving_path = save_dir+"/maps"
    metrics_saving_path = save_dir+"/metrics"
    os.makedirs(ts_saving_path)
    os.makedirs(map_saving_path)
    os.makedirs(metrics_saving_path)


    dataset = dataset_ST_MultiPoint.Dataset_ST_MultiPoint(config)
    
    models_predictions = {}
    
    Z_grid = plot_ST_MultiPoint.grid_generation(dataset,
                                                config["lat_lon_npoints"][0],
                                                config["lat_lon_npoints"][1])
    
    # Load the model and compute predictions
    for i in range(len(config["model_name"])):
        
        model_config = {}
        with open(config["model_config_path"][i]) as f:
            model_config = json.load(f)
            print(f"Read data.json: {config['model_config_path'][i]}")
        
        model = load_model(model_config, config["model_path"][i])
        model = model.to(device)
        print("Total number of trainable parameters: " ,sum(p.numel() for p in model.parameters() if p.requires_grad and p != "Densification_Dropout"))
        
        if config["get_displacement"] and config["model_name"][i] in config["model_with_displacements"]:
            
            models_predictions[config["model_name"][i]] = compute_prediction_with_displacement(config, dataset, device,
                                  model,
                                  config["iter_pred"],
                                  Z_grid)
            
        else:
            
            models_predictions[config["model_name"][i]] = compute_prediction(config, dataset, device,
                        model,
                        config["iter_pred"],
                        Z_grid)
            
    # Compute metrics
    if config["n_pred_ts"]>0:
        
        if config["compute_metrics"] is True:
            
            print("Computing metrics...", end = " ")
            median_metrics_dict = {}
            mean_metrics_dict = {}
            std_metrics_dict = {}
            
            name_suffix = ""
            
            if config["iter_pred"]:
                name_suffix += "_iter_pred"
                
            if config["forecast_horizon"] is not None:
                name_suffix += f"_FO{config['forecast_horizon']}"
                
            #Compute denormalized sensor means
            subset_wtd_df = dataset.wtd_df.loc[pd.IndexSlice[dataset.wtd_df.index.get_level_values(0) <= np.datetime64(dataset.config["date_max_norm"]),
                                                            :]] #
            
            subset_wtd_df = (subset_wtd_df[dataset.target] * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
            
            sensor_means = subset_wtd_df.groupby(level=1).transform('mean').values
            sensor_means = sensor_means.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list),
                                                len(dataset.sensor_id_list))[0,:] 
            
            sensor_min = subset_wtd_df.groupby(level=1).transform('min').values
            sensor_min = sensor_min.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list),
                                                len(dataset.sensor_id_list))[0,:] 
            
            sensor_max = subset_wtd_df.groupby(level=1).transform('max').values
            sensor_max = sensor_max.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list),
                                                len(dataset.sensor_id_list))[0,:] 
            sensor_iv = sensor_max - sensor_min
        
                
            for model_i in config["model_name"]:

                model_median_metrics = []
                model_mean_metrics = []
                model_std_metrics = []
                
                if config["metrics_only_on_test"] is True:
                    true_values = models_predictions[model_i][0][0].loc[models_predictions[model_i][0][0].index>np.datetime64(config["test_split_p"])]
                    predicted_values = models_predictions[model_i][0][1].loc[models_predictions[model_i][0][0].index>np.datetime64(config["test_split_p"])]
                else:
                    true_values = models_predictions[model_i][0][0]
                    predicted_values = models_predictions[model_i][0][1]
                    
                sensors_nbias = metrics.compute_test_nbias_per_sensor(true_values,
                                                                    predicted_values,
                                                                    sensor_iv)
                sensors_nbias.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_nbias.csv", index=True)
                model_median_metrics.append(sensors_nbias.median())
                model_mean_metrics.append(sensors_nbias.mean())
                model_std_metrics.append(sensors_nbias.std())
                
                sensors_rmse = metrics.compute_test_rmse_per_sensor(true_values,
                                                                    predicted_values)
                sensors_rmse.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_rmse.csv", index=True)
                model_median_metrics.append(sensors_rmse.median())
                model_mean_metrics.append(sensors_rmse.mean())
                model_std_metrics.append(sensors_rmse.std())
                
                sensors_mape = metrics.compute_test_mape_per_sensor(true_values,
                                                                    predicted_values)
                sensors_mape.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_mape.csv", index=True)
                model_median_metrics.append(sensors_mape.median())
                model_mean_metrics.append(sensors_mape.mean())
                model_std_metrics.append(sensors_mape.std())
                
                sensors_nse = metrics.compute_test_nse_per_sensor(true_values,
                                                                predicted_values,
                                                                sensor_means)
                sensors_nse.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_nse.csv", index=True)
                model_median_metrics.append(sensors_nse.median())
                model_mean_metrics.append(sensors_nse.mean())
                model_std_metrics.append(sensors_nse.std())
                
                sensors_kge = metrics.compute_test_kge_per_sensor(true_values,
                                                                predicted_values)
                sensors_kge.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_kge.csv", index=True)
                model_median_metrics.append(sensors_kge.median())
                model_mean_metrics.append(sensors_kge.mean())
                model_std_metrics.append(sensors_kge.std())
                
                median_metrics_dict[model_i] = model_median_metrics
                mean_metrics_dict[model_i] = model_mean_metrics
                std_metrics_dict[model_i] = model_std_metrics
                
            median_metrics_ds = pd.DataFrame(median_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
            mean_metrics_ds = pd.DataFrame(mean_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
            std_metrics_ds = pd.DataFrame(std_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
                
            median_metrics_ds.to_csv(f"{metrics_saving_path}/median_metrics{name_suffix}.csv", index=True)
            mean_metrics_ds.to_csv(f"{metrics_saving_path}/mean_metrics{name_suffix}.csv", index=True)
            std_metrics_ds.to_csv(f"{metrics_saving_path}/std_metrics{name_suffix}.csv", index=True)
            print("Saved!")
            
        # Time Series plot
        print("Drawing plots...")
        for sensor_idx in range(len(dataset.sensor_id_list)):

            sensor = dataset.sensor_id_list[sensor_idx]
            munic = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor,"munic"].values[0]

            fig, ax = plt.subplots(1,1, figsize = (13,3)) #(12,5)
            plt.title(f"{munic} - {sensor}")
            
            markers = ['s', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'd', 'H', '|', '_']
            colors = config["ts_colors"] #['tab:brown','tab:orange','darkgreen','darkmagenta']
            i = 0
            for model_i in config["model_name"]:
                
                if config["forecast_horizon"] is not None:
                    for j in range(config["n_pred_ts"]):
                        models_predictions[model_i][0][1][sensor].iloc[j*config["forecast_horizon"]:(j+1)*config["forecast_horizon"]].plot(
                                                                                            ax = ax,
                                                                                            color = colors[i % len(markers)],
                                                                                            marker=markers[i % len(markers)],
                                                                                            label = f"{model_i}" if j == config["n_pred_ts"]-1 else "",
                                                                                            #markersize = 2.5, linewidth = 0.8,
                                                                                            markersize = 1.5, linewidth = 0.2,
                                                                                            )
                else:
                    models_predictions[model_i][0][1][sensor].plot(label = f"{model_i}", ax = ax,
                                                            color = colors[i % len(markers)],
                                                            marker=markers[i % len(markers)],
                                                            #markersize = 2.5, linewidth = 0.8
                                                            markersize = 1.5, linewidth = 0.2
                                                            )
                
                if model_i == config["model_name"][-1] :
                    models_predictions[model_i][0][0][sensor].plot(label = "Truth", ax = ax,
                                                                color = "tab:blue",
                                                                marker = "o", linestyle = "--" ,
                                                                #markersize = 4, linewidth = 2
                                                                markersize = 1.5, linewidth = 0.2
                                                                )
                    
                
                i += 1
                
            ax.set_ylim([ax.get_ylim()[0] - ax.get_ylim()[0]*0.0005,
                    ax.get_ylim()[1] + ax.get_ylim()[1].min()*0.0005])
            
            # Start Test Vline
            ax.vlines(config["test_split_p"], ymin = ax.get_ylim()[0],
                    ymax = ax.get_ylim()[1], ls = "--", color = "darkred", lw = 2,
                    label = "Start Test")
            
            # Grey boxes for missing values
            all_dates = models_predictions[model_i][0][0][sensor].index.get_level_values(0)
            if (models_predictions[model_i][0][0][sensor].isnull().any()):
                ax.bar(all_dates[models_predictions[model_i][0][0][sensor].isnull()],
                        bottom = ax.get_ylim()[0],
                        height = ax.get_ylim()[1],
                        width= 2,
                        align='center',
                        color = 'lightgrey',
                        label = "Missing Values", zorder = 0)
            
            print(f"Saving Time Series of {munic} - {sensor}")
            plt.xlabel("Date")
            plt.ylabel("Groundwater Level [m]")
            plt.legend(ncol=len(plt.gca().get_legend_handles_labels()[0]))
            ax.grid(axis="x", ls = "--", which = "both", lw = "1.5")
            
            # for all ds dates 
            date_xticks = pd.date_range(np.datetime64("2001-01-01"), np.datetime64("2023-12-31"), freq = "6MS",  normalize = True, inclusive = "both")
            ax.set_xticks(date_xticks, date_xticks.strftime('%d/%m/%Y'))
            ax.tick_params(axis = "x", rotation=50)
            
            if config["forecast_horizon"] is None:
                n_pred = config['n_pred_ts']
            else:
                n_pred = config['n_pred_ts']*config["forecast_horizon"]
            title = f"{ts_saving_path}/{munic}_{sensor}_{config['start_date_pred_ts']}_{n_pred}"
            
            if config["iter_pred"]:
                title += "_iter_pred"
                
            if config["forecast_horizon"] is not None:
                title += f"_FO{config['forecast_horizon']}"
                
            plt.savefig(f"{title}.png", bbox_inches='tight', dpi=400, pad_inches=0.1) #dpi = 400, transparent = True
            plt.close("all")
                
                
        print("All time series plots saved!")
    
    if config["n_pred_map"]>0:
        
        print("Drawing maps...", end = " ")
        for date in config["map_dates"]:
            
            save_map_dir = f"{map_saving_path}/maps_{date.replace('-','_')}"
            
            if config["iter_pred"]:
                save_map_dir += "_iter_pred"
                
            if config["forecast_horizon"] is not None:
                save_map_dir += f"_FO{config['forecast_horizon']}"
            
            ### Map Plots H 
            model_pred_list_H = [models_predictions[config["model_name"][i]][1][0].sel(time = date) for i in range(len(config["model_name"]))]
            model_pred_list_WTD = [models_predictions[config["model_name"][i]][1][1].sel(time = date) for i in range(len(config["model_name"]))]
        
            plot_ST_MultiPoint.plot_map_all_models(model_pred_list_H,
                title = f"{date} Predictions Groundwater Level",
                shapefile = dataset.piemonte_shp,
                model_names = config["model_name"],
                cmap = "Blues",
                var_name_title = "GWL [m]",
                save_dir = save_map_dir + "_GWL", 
                print_plot = False)
            plt.close("all")
            
            ### Map Plots WTD
            
            plot_ST_MultiPoint.plot_map_all_models(model_pred_list_WTD,
                title = f"{date} Predictions Water Table Depth",
                shapefile = dataset.piemonte_shp,
                model_names = config["model_name"],
                cmap = "Blues_r",
                var_name_title = "WTD [m]",
                save_dir = save_map_dir + "_WTD", 
                print_plot = False)
            plt.close("all")
            ### Map Plots Displacements
            model_pred_displacements_list = [] 
            
            for i in range(len(config["model_with_displacements"])):
                for j in range(3):
                    
                    displacement_list = []
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][2].sel(time = date))
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][3].sel(time = date))
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][4].sel(time = date))
                    
                model_pred_displacements_list.append(displacement_list)
            
            plot_ST_MultiPoint.plot_displacement_all_models(model_pred_displacements_list,
                title = f"{date} Predicted Displacements",
                shapefile = dataset.piemonte_shp,
                recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                model_names = config["model_with_displacements"],
                save_dir = save_map_dir + "_Deltas", 
                print_plot = False)
            plt.close("all")
        
        print("All Maps saved!")
        #######
        # Gif #
        #######
        
        print("Drawing GIFs...", end = " ")
        save_gif_dir = f"{map_saving_path}/gif_from_{config['start_date_pred_map'].replace('-','_')}"
            
        if config["iter_pred"]:
            save_gif_dir += "_iter_pred"
            
        if config["forecast_horizon"] is not None:
            save_gif_dir += f"_FO{config['forecast_horizon']}"
        
        ### H
        for model in config["model_name"]:
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][0],
                            title = f"{model} - Groundwater Level [m] Evolution",
                            shapefile = dataset.piemonte_shp,
                            freq = "W",
                            cmap = "Blues",
                            vmin_1 = False,
                            vmax_1 = False,
                            save_dir = save_gif_dir + f"_GWL_{model}",
                            print_plot = False)
            
            plt.close("all")
            
        
            
        for model in config["model_with_displacements"]:
            ### Delta GW
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][2],
                            title = r"{} $\Delta_{{GW}}$ [m] Evolution".format(model),
                            shapefile = dataset.piemonte_shp,
                            recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                            freq = "W",
                            cmap = "seismic_r",
                            save_dir = save_gif_dir + f"_DGW_{model}",
                            print_plot = False)
            plt.close("all")
        
        
            ### Delta S
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][3],
                            title = r"{} $\Delta_S$ [m] Evolution".format(model),
                            shapefile = dataset.piemonte_shp,
                            recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                            freq = "W",
                            cmap = "seismic_r",
                            save_dir = save_gif_dir + f"_DS_{model}",
                            print_plot = False)
            plt.close("all")
            
            print("All GIFs saved!")
      

if __name__ == "__main__":
    args = parse_arguments()

    config = {}
    with open(args.config) as f:
        config = json.load(f)
    
    if config["stdout_log_dir"] is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_names_dir = "_".join(config["model_name"])
        save_dir_stdout = '{}_{}_{}.txt'.format(config["stdout_log_dir"],model_names_dir,timestamp)
        save_dir_stderr = '{}_{}_{}.txt'.format(config["stderr_log_dir"],model_names_dir,timestamp)
            
        # Redirect sys.stdout and err to the files
        sys.stdout = open(save_dir_stdout, 'w')
        sys.stderr = open(save_dir_stderr, 'w')

    main(config)