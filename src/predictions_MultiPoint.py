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
    os.makedirs(ts_saving_path)
    os.makedirs(map_saving_path)


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
            
    # Time Series plot
    
    print("Drawing plots...")
    for sensor_idx in range(len(dataset.sensor_id_list)):

        sensor = dataset.sensor_id_list[sensor_idx]
        munic = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor,"munic"].values[0]

        fig, ax = plt.subplots(1,1, figsize = (12,5))
        plt.title(f"{munic} - {sensor}")
        
        markers = ['s', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'd', 'H', '|', '_']
        i = 0
        for model_i in config["model_name"]:
             
            models_predictions[model_i][0][1][sensor].plot(label = f"{model_i}", ax = ax,
                                                    marker=markers[i % len(markers)], markersize = 2, linewidth = 0.8)
            
            if model_i == config["model_name"][-1] :
                models_predictions[model_i][0][0][sensor].plot(label = "Truth", ax = ax,
                                                            color = "black",
                                                            marker = "o", linestyle = "--" , markersize = 4, linewidth = 2)
            i += 1
        print(f"Saving Time Series of {munic} - {sensor}")
        plt.xlabel("Date")
        plt.ylabel("H [m]")
        plt.legend()
        ax.grid(axis="x", ls = "--", which = "both", lw = "1.5")
        title = f"{ts_saving_path}/{munic}_{sensor}_{config['start_date_pred_ts']}_{config['n_pred_ts']}"
        
        if config["iter_pred"]:
            title += "_iter_pred"
            
        plt.savefig(f"{title}.png", bbox_inches='tight', dpi=400, pad_inches=0.1) #dpi = 400, transparent = True
        plt.close("all")
            
            
    print("All time series plots saved!")
    
    print("Drawing maps...")
    for date in config["map_dates"]:
        
        save_map_dir = f"{map_saving_path}/maps_{date.replace('-','_')}"
        
        if config["iter_pred"]:
            save_map_dir += "_iter_pred"
        
        ### Map Plots H 
        model_pred_list_H = [models_predictions[config["model_name"][i]][1][0].sel(time = date) for i in range(len(config["model_name"]))]
        model_pred_list_WTD = [models_predictions[config["model_name"][i]][1][1].sel(time = date) for i in range(len(config["model_name"]))]
    
        plot_ST_MultiPoint.plot_map_all_models(model_pred_list_H,
            title = f"{date} Predictions Piezometric Head",
            shapefile = dataset.piemonte_shp,
            model_names = config["model_name"],
            var_name_title = "H [m]",
            save_dir = save_map_dir + "_H", 
            print_plot = False)
        plt.close("all")
        
        ### Map Plots WTD
        
        plot_ST_MultiPoint.plot_map_all_models(model_pred_list_WTD,
            title = f"{date} Predictions Water Table Depth",
            shapefile = dataset.piemonte_shp,
            model_names = config["model_name"],
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
    
    print("Drawing GIFs...")
    save_gif_dir = f"{map_saving_path}/gif_from_{config['start_date_pred_map'].replace('-','_')}"
        
    if config["iter_pred"]:
        save_gif_dir += "_iter_pred"
    
    ### H
    for model in config["model_name"]:
        plot_ST_MultiPoint.generate_gif_from_xr(date, config["n_pred_ts"],
                        models_predictions[model][1][0],
                        title = f"{model} - Piezometric Head [m] Evolution",
                        shapefile = dataset.piemonte_shp,
                        freq = "W",
                        cmap = "viridis",
                        save_dir = save_gif_dir + f"_H_{model}",
                        print_plot = False)
        
        plt.close("all")
        
    
        
    for model in config["model_with_displacements"]:
        ### Delta GW
        plot_ST_MultiPoint.generate_gif_from_xr(date, config["n_pred_ts"],
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
        plot_ST_MultiPoint.generate_gif_from_xr(date, config["n_pred_ts"],
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
    
    
    
            
            
    
def load_model(config, model_path):
    model, _ = load_model_ST_MultiPoint.load_model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    
    return model

### Prediction Functions
def compute_ts_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred):
    ### Sensors Time Series Prediction
        
        ts_true, ts_predictions, ts_Displacement_GW, ts_Displacement_S, ts_Conductivity, ts_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                    np.datetime64(config["start_date_pred_ts"]),
                                                                                    config["n_pred_ts"],
                                                                                    iter_pred = iter_pred,
                                                                                    get_displacement_terms = True)
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_GW_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_GW.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_S_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_S.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Conductivity_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Conductivity.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Lag_GW = plot_ST_MultiPoint.build_ds_from_pred(ts_Lag_GW.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        
        
        # Denormalize
        ts_true_ds = (ts_true_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        ts_predictions_ds = (ts_predictions_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        ts_Displacement_GW_ds = ts_Displacement_GW_ds * dataset.norm_factors["target_stds"]
        ts_Displacement_S_ds = ts_Displacement_S_ds * dataset.norm_factors["target_stds"]
        ts_Conductivity_ds = ts_Conductivity_ds * dataset.norm_factors["target_stds"]
        ts_Lag_GW = (ts_Lag_GW * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        
        return [ts_true_ds, ts_predictions_ds,
                ts_Displacement_GW_ds, ts_Displacement_S_ds,
                ts_Conductivity_ds, ts_Lag_GW]
        
def compute_ts_prediction(config, dataset,
                        model, device,
                        iter_pred):
    ### Sensors Time Series Prediction
        
        ts_true, ts_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                    np.datetime64(config["start_date_pred_ts"]),
                                                                                    config["n_pred_ts"],
                                                                                    iter_pred = iter_pred,
                                                                                    get_displacement_terms = False)
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        # Denormalize
        ts_true_ds = (ts_true_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        ts_predictions_ds = (ts_predictions_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
       
        return [ts_true_ds, ts_predictions_ds]
        
def compute_grid_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred,
                                           Z_grid):
    
    ### ST Grid prediction
        _, grid_predictions, grid_Displacement_GW, grid_Displacement_S, grid_Conductivity, grid_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                                                                                                    np.datetime64(config["start_date_pred_map"]),
                                                                                                                                                                    config["n_pred_map"],
                                                                                                                                                                    iter_pred = iter_pred,
                                                                                                                                                                    get_displacement_terms = True,
                                                                                                                                                                    Z_grid = Z_grid)
        
        grid_predictions = grid_predictions.reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Displacement_GW = grid_Displacement_GW.reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Displacement_S = grid_Displacement_S.reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Conductivity = grid_Conductivity.reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Lag_GW = grid_Lag_GW.reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        
        grid_predictions = (grid_predictions * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        grid_Lag_GW = (grid_Lag_GW * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        grid_Displacement_GW = grid_Displacement_GW * dataset.norm_factors["target_stds"]
        grid_Displacement_S = grid_Displacement_S * dataset.norm_factors["target_stds"]
        grid_Conductivity = grid_Conductivity * dataset.norm_factors["target_stds"]
                                                                                                                                                              
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred_map"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(config["n_pred_map"])]

        Z_grid_matrix = Z_grid.reshape(config["lat_lon_npoints"][0],config["lat_lon_npoints"][1],3)
        Z_grid_matrix_lat = (Z_grid_matrix[:,:,0] * dataset.norm_factors["lat_std"]) + dataset.norm_factors["lat_mean"]
        Z_grid_matrix_lon = (Z_grid_matrix[:,:,1] * dataset.norm_factors["lon_std"]) + dataset.norm_factors["lon_mean"]
        dtm = (Z_grid_matrix[:,:,2] * dataset.norm_factors["dtm_std"].values) + dataset.norm_factors["dtm_mean"].values
                
        predictions_xr = xarray.DataArray(data = grid_predictions,
                                        coords = dict(
                                                    lat=("lat", Z_grid_matrix_lat[:,0]),
                                                    lon=("lon", Z_grid_matrix_lon[0,:]),
                                                    time=date_seq),
                                        dims = ["time","lat", "lon"]
                                        )
        
        predictions_wtd_xr = dtm - predictions_xr
            
        displacement_gw_xr = xarray.DataArray(data = grid_Displacement_GW,
                            coords = dict(
                                        lat=("lat", Z_grid_matrix_lat[:,0]),
                                        lon=("lon", Z_grid_matrix_lon[0,:]),
                                        time=date_seq),
                            dims = ["time","lat", "lon"]
                            )

        displacement_s_xr = xarray.DataArray(data = grid_Displacement_S,
                            coords = dict(
                                        lat=("lat", Z_grid_matrix_lat[:,0]),
                                        lon=("lon", Z_grid_matrix_lon[0,:]),
                                        time=date_seq),
                            dims = ["time","lat", "lon"]
                            )

        conductivity_xr = xarray.DataArray(data = grid_Conductivity,
                            coords = dict(
                                        lat=("lat", Z_grid_matrix_lat[:,0]),
                                        lon=("lon", Z_grid_matrix_lon[0,:]),
                                        time=date_seq),
                            dims = ["time","lat", "lon"]
                            )
        
        Lag_GW_xr = xarray.DataArray(data = grid_Lag_GW,
                            coords = dict(
                                        lat=("lat", Z_grid_matrix_lat[:,0]),
                                        lon=("lon", Z_grid_matrix_lon[0,:]),
                                        time=date_seq),
                            dims = ["time","lat", "lon"]
                            )
        
        return [predictions_xr, predictions_wtd_xr,
                displacement_gw_xr,displacement_s_xr,
                conductivity_xr, Lag_GW_xr]
        
def compute_grid_prediction(config, dataset,
                            model, device,
                            iter_pred,
                            Z_grid):
    
    ### ST Grid prediction
        _, grid_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                np.datetime64(config["start_date_pred_map"]),
                                                                                config["n_pred_map"],
                                                                                iter_pred = iter_pred,
                                                                                get_displacement_terms = False,
                                                                                Z_grid = Z_grid)
        
        grid_predictions = grid_predictions.detach().cpu().reshape(config["n_pred_map"],config["lat_lon_npoints"][0],config["lat_lon_npoints"][1])
        
        grid_predictions = (grid_predictions * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
                                                                                                                                                            
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred_map"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(config["n_pred_map"])]

        Z_grid_matrix = Z_grid.reshape(config["lat_lon_npoints"][0],config["lat_lon_npoints"][1],3)
        Z_grid_matrix_lat = (Z_grid_matrix[:,:,0] * dataset.norm_factors["lat_std"]) + dataset.norm_factors["lat_mean"]
        Z_grid_matrix_lon = (Z_grid_matrix[:,:,1] * dataset.norm_factors["lon_std"]) + dataset.norm_factors["lon_mean"]
        dtm = (Z_grid_matrix[:,:,2] * dataset.norm_factors["dtm_std"].values) + dataset.norm_factors["dtm_mean"].values
                
        predictions_xr = xarray.DataArray(data = grid_predictions,
                                        coords = dict(
                                                    lat=("lat", Z_grid_matrix_lat[:,0]),
                                                    lon=("lon", Z_grid_matrix_lon[0,:]),
                                                    time=date_seq),
                                        dims = ["time","lat", "lon"]
                                        )
        
        predictions_wtd_xr = dtm - predictions_xr
        
        return [predictions_xr, predictions_wtd_xr]

### General Functions
def compute_prediction_with_displacement(config, dataset, device,
                                  model,
                                  iter_pred,
                                  Z_grid):
    with torch.no_grad():
        model.eval()
        print("Computing Time Series Predictions...")
        ts_true_ds, ts_predictions_ds, ts_Displacement_GW_ds, ts_Displacement_S_ds, ts_Conductivity_ds, ts_Lag_GW_dict = compute_ts_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred)
        print("Done!")
        print("Computing Gridded Predictions...")
        predictions_xr, predictions_wtd_xr, displacement_gw_xr,displacement_s_xr, conductivity_xr, Lag_GW_xr = compute_grid_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred,
                                           Z_grid)
        print("Done!")
        
        return [[ts_true_ds, ts_predictions_ds, ts_Displacement_GW_ds, ts_Displacement_S_ds, ts_Conductivity_ds, ts_Lag_GW_dict],
                [predictions_xr, predictions_wtd_xr, displacement_gw_xr,displacement_s_xr, conductivity_xr, Lag_GW_xr]]     
        
def compute_prediction(config, dataset, device,
                        model,
                        iter_pred,
                        Z_grid):
    with torch.no_grad():
        model.eval()
        print("Computing Time Series Predictions...")
        ts_true_ds, ts_predictions_ds = compute_ts_prediction(config, dataset,
                            model, device,
                            iter_pred)
        print("Done!")
        print("Computing Gridded Predictions...")
        predictions_xr, predictions_wtd_xr = compute_grid_prediction(config, dataset,
                                model, device,
                                iter_pred,
                                Z_grid)
        print("Done!")
        
        return [[ts_true_ds, ts_predictions_ds],
                [predictions_xr, predictions_wtd_xr]]
        

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