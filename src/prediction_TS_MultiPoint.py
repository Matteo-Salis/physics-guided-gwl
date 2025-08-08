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
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print("Device: ", device)


    dataset = dataset_ST_MultiPoint.Dataset_ST_MultiPoint(config)
    
    models_predictions = {}
    
    Z_grid = plot_ST_MultiPoint.grid_generation(dataset,
                                                config["lat_points"],
                                                config["lon_points"])
    
    # Load the model and compute predictions
    for i in range(len(config["model_name"])):
        
        model_config = {}
        with open(config["model_config_path"][i]) as f:
            model_config = json.load(f)
            print(f"Read data.json: {config['model_config_path'][i]}")
        
        model = load_model(model_config, config["model_path"][i])
        model = model.to(device)
        print("Total number of trainable parameters: " ,sum(p.numel() for p in model.parameters() if p.requires_grad and p != "Densification_Dropout"))
        
        if config["get_displacement"]:
            
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
                                                    marker=markers[i % len(markers)], markersize = 3, linewidth = 0.8)
            
            if model_i == config["model_name"][-1] :
                models_predictions[model_i][0][0][sensor].plot(label = "Truth", ax = ax,
                                                            color = "black",
                                                            marker = "o", linestyle = "--" , markersize = 5, linewidth = 2)
            i += 1
        print(f"Saving Time Series of {munic} - {sensor}")
        plt.legend()
        title = f"{config['time_series_saving_path']}/{munic}_{sensor}_{config['start_date_pred']}_{config['n_pred_ts']}"
        
        if config["iter_pred"]:
            title += "_iter_pred"
            
        plt.savefig(f"{title}.png", bbox_inches='tight', dpi=400, pad_inches=0.1) #dpi = 400, transparent = True
            
            
    print("END: All plots saved!")
    
    # Gif
            
            
    
def load_model(config, model_path):
    model, _ = load_model_ST_MultiPoint.load_model(config)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    
    return model
    
def compute_ts_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred):
    ### Sensors Time Series Prediction
        
        ts_true, ts_predictions, ts_Displacement_GW, ts_Displacement_S, ts_Permeability, ts_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                    np.datetime64(config["start_date_pred"]),
                                                                                    config["n_pred_ts"],
                                                                                    iter_pred = iter_pred,
                                                                                    get_displacement_terms = True)
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_GW_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_GW, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_S_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_S, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Permeability_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Permeability, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Lag_GW_dict = {f"ts_Lag_GW{i}": plot_ST_MultiPoint.build_ds_from_pred(ts_Lag_GW[:,i,:], dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list) for i in config["target_lags"]}
        
        
        
        # Denormalize
        ts_true_ds = (ts_true_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        ts_predictions_ds = (ts_predictions_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        ts_Displacement_GW_ds = ts_Displacement_GW_ds * dataset.norm_factors["target_stds"]
        ts_Displacement_S_ds = ts_Displacement_S_ds * dataset.norm_factors["target_stds"]
        ts_Permeability_ds = ts_Permeability_ds * dataset.norm_factors["target_stds"]
        ts_Lag_GW_dict = {lag: (ts_Lag_GW_dict[lag] * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"] for lag in list(ts_Lag_GW_dict.keys())}
        
        return [ts_true_ds, ts_predictions_ds,
                ts_Displacement_GW_ds, ts_Displacement_S_ds,
                ts_Permeability_ds, ts_Lag_GW_dict]
        
def compute_ts_prediction(config, dataset,
                        model, device,
                        iter_pred):
    ### Sensors Time Series Prediction
        
        ts_true, ts_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                    np.datetime64(config["start_date_pred"]),
                                                                                    config["n_pred_ts"],
                                                                                    iter_pred = iter_pred,
                                                                                    get_displacement_terms = False)
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions, dataset,
                                                        start_date=np.datetime64(config["start_date_pred"]), n_pred=config["n_pred_ts"],
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
        _, grid_predictions, grid_Displacement_GW, grid_Displacement_S, grid_Permeability, grid_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                                                                                                    np.datetime64(config["start_date_pred"]),
                                                                                                                                                                    config["n_pred_ts"],
                                                                                                                                                                    iter_pred = iter_pred,
                                                                                                                                                                    get_displacement_terms = True,
                                                                                                                                                                    Z_grid = Z_grid)
        
        grid_predictions = grid_predictions.reshape(config["n_pred_ts"],config["lat_points"],config["lon_points"])
        grid_Displacement_GW = grid_Displacement_GW.reshape(config["n_pred_ts"],config["lat_points"],config["lon_points"])
        grid_Displacement_S = grid_Displacement_S.reshape(config["n_pred_ts"],config["lat_points"],config["lon_points"])
        grid_Permeability = grid_Permeability.reshape(config["n_pred_ts"],config["lat_points"],config["lon_points"])
        grid_Lag_GW = grid_Lag_GW.reshape(config["n_pred_ts"],config["target_lags"],config["lat_points"],config["lon_points"])
                                                                                                                                                                    
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(config["n_pred_ts"])]

        Z_grid_matrix = Z_grid.reshape(config["lat_points"],config["lon_points"],3)
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

        permeability_xr = xarray.DataArray(data = grid_Permeability,
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
                                        time=date_seq,
                                        lag=config["target_lags"]),
                            dims = ["time","lag","lat", "lon"]
                            )
        
        return [predictions_xr, predictions_wtd_xr,
                displacement_gw_xr,displacement_s_xr,
                permeability_xr, Lag_GW_xr]
        
        
def compute_grid_prediction(config, dataset,
                            model, device,
                            iter_pred,
                            Z_grid):
    
    ### ST Grid prediction
        _, grid_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                np.datetime64(config["start_date_pred"]),
                                                                                config["n_pred_ts"],
                                                                                iter_pred = iter_pred,
                                                                                get_displacement_terms = False,
                                                                                Z_grid = Z_grid)
        
        grid_predictions = grid_predictions.reshape(config["n_pred_ts"],config["lat_points"],config["lon_points"])
                                                                                                                                                                    
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(config["n_pred_ts"])]

        Z_grid_matrix = Z_grid.reshape(config["lat_points"],config["lon_points"],3)
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

def compute_prediction_with_displacement(config, dataset, device,
                                  model,
                                  iter_pred,
                                  Z_grid):

        print("Computing Time Series Predictions...")
        ts_true_ds, ts_predictions_ds, ts_Displacement_GW_ds, ts_Displacement_S_ds, ts_Permeability_ds, ts_Lag_GW_dict = compute_ts_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred)
        print("Done!")
        print("Computing Gridded Predictions...")
        predictions_xr, predictions_wtd_xr, displacement_gw_xr,displacement_s_xr, permeability_xr, Lag_GW_xr = compute_grid_prediction_with_displacemnt(config, dataset,
                                           model, device,
                                           iter_pred,
                                           Z_grid)
        print("Done!")
        
        return [[ts_true_ds, ts_predictions_ds, ts_Displacement_GW_ds, ts_Displacement_S_ds, ts_Permeability_ds, ts_Lag_GW_dict],
                [predictions_xr, predictions_wtd_xr, displacement_gw_xr,displacement_s_xr, permeability_xr, Lag_GW_xr]]
        
        
def compute_prediction(config, dataset, device,
                        model,
                        iter_pred,
                        Z_grid):

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

    main(config)