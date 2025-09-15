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
        if config["forecast_horizon"] is None:
            
            ts_true, ts_predictions, ts_Displacement_GW, ts_Displacement_S, ts_Conductivity, ts_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                        np.datetime64(config["start_date_pred_ts"]),
                                                                                        config["n_pred_ts"],
                                                                                        iter_pred = iter_pred,
                                                                                        get_displacement_terms = True)
            
            n_pred = config["n_pred_ts"]
            
        else:
            ts_true_list = []
            ts_predictions_list = []
            ts_Displacement_GW_list = []
            ts_Displacement_S_list = []
            ts_Conductivity_list = []
            ts_Lag_GW_list = []
            
            date = np.datetime64(config["start_date_pred_ts"])
            
            for i in range(config["n_pred_ts"]):
                
                ts_true, ts_predictions, ts_Displacement_GW, ts_Displacement_S, ts_Conductivity, ts_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                        date,
                                                                                        config["forecast_horizon"],
                                                                                        iter_pred = iter_pred,
                                                                                        get_displacement_terms = True)
                ts_true_list.append(ts_true)
                ts_predictions_list.append(ts_predictions)
                ts_Displacement_GW_list.append(ts_Displacement_GW)
                ts_Displacement_S_list.append(ts_Displacement_S)
                ts_Conductivity_list.append(ts_Conductivity)
                ts_Lag_GW_list.append(ts_Lag_GW)
                
                date = date + np.timedelta64(config["forecast_horizon"], config["frequency"])
                
            ts_true = torch.cat(ts_true_list, dim = 0)
            ts_predictions = torch.cat(ts_predictions_list, dim = 0)
            ts_Displacement_GW = torch.cat(ts_Displacement_GW_list, dim = 0)
            ts_Displacement_S = torch.cat(ts_Displacement_S_list, dim = 0)
            ts_Conductivity = torch.cat(ts_Conductivity_list, dim = 0)
            ts_Lag_GW = torch.cat(ts_Lag_GW_list, dim = 0)
            
            n_pred = config["forecast_horizon"] * config["n_pred_ts"]
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_GW_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_GW.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Displacement_S_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Displacement_S.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Conductivity_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_Conductivity.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_Lag_GW = plot_ST_MultiPoint.build_ds_from_pred(ts_Lag_GW.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
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
        
        if config["forecast_horizon"] is None:
            ts_true, ts_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                        np.datetime64(config["start_date_pred_ts"]),
                                                                                        config["n_pred_ts"],
                                                                                        iter_pred = iter_pred,
                                                                                        get_displacement_terms = False)
            
            n_pred = config["n_pred_ts"]
            
        else:
            ts_true_list = []
            ts_predictions_list = []
            
            date = np.datetime64(config["start_date_pred_ts"])
            
            for i in range(config["n_pred_ts"]):
                
                ts_true, ts_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                        date,
                                                                                        config["forecast_horizon"],
                                                                                        iter_pred = iter_pred,
                                                                                        get_displacement_terms = False)
                ts_true_list.append(ts_true)
                ts_predictions_list.append(ts_predictions)
                
                date = date + np.timedelta64(config["forecast_horizon"], config["frequency"])
                
            ts_true = torch.cat(ts_true_list, dim = 0)
            ts_predictions = torch.cat(ts_predictions_list, dim = 0)
            
            n_pred = config["forecast_horizon"] * config["n_pred_ts"]
        
        # Build Pandas DF
        
        ts_true_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_true.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
                                                        sensor_names=dataset.sensor_id_list)
        
        ts_predictions_ds = plot_ST_MultiPoint.build_ds_from_pred(ts_predictions.detach().cpu(), dataset,
                                                        start_date=np.datetime64(config["start_date_pred_ts"]), n_pred=n_pred,
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
        if config["forecast_horizon"] is None:
        
            _, grid_predictions, grid_Displacement_GW, grid_Displacement_S, grid_Conductivity, grid_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                                                                                                        np.datetime64(config["start_date_pred_map"]),
                                                                                                                                                                        config["n_pred_map"],
                                                                                                                                                                        iter_pred = iter_pred,
                                                                                                                                                                        get_displacement_terms = True,
                                                                                                                                                                        Z_grid = Z_grid)
            
            n_pred = config["n_pred_map"]
            
        else:
            
            grid_predictions_list = []
            grid_Displacement_GW_list = []
            grid_Displacement_S_list = []
            grid_Conductivity_list = []
            grid_Lag_GW_list = []
            
            date = np.datetime64(config["start_date_pred_ts"])
            
            for i in range(config["n_pred_map"]):
                _, grid_predictions, grid_Displacement_GW, grid_Displacement_S, grid_Conductivity, grid_Lag_GW = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                                                                                                        date,
                                                                                                                                                                        config["forecast_horizon"],
                                                                                                                                                                        iter_pred = iter_pred,
                                                                                                                                                                        get_displacement_terms = True,
                                                                                                                                                                        Z_grid = Z_grid)
                
                grid_predictions_list.append(grid_predictions)
                grid_Displacement_GW_list.append(grid_Displacement_GW)
                grid_Displacement_S_list.append(grid_Displacement_S)
                grid_Conductivity_list.append(grid_Conductivity)
                grid_Lag_GW_list.append(grid_Lag_GW)
                
                date = date + np.timedelta64(config["forecast_horizon"], config["frequency"])
            
            grid_predictions = torch.cat(grid_predictions_list, dim = 0)
            grid_Displacement_GW = torch.cat(grid_Displacement_GW_list, dim = 0)
            grid_Displacement_S = torch.cat(grid_Displacement_S_list, dim = 0)
            grid_Conductivity = torch.cat(grid_Conductivity_list, dim = 0)
            grid_Lag_GW = torch.cat(grid_Lag_GW_list, dim = 0)
            
            n_pred = config["forecast_horizon"] * config["n_pred_map"]
        
                       
        grid_predictions = grid_predictions.reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Displacement_GW = grid_Displacement_GW.reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Displacement_S = grid_Displacement_S.reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Conductivity = grid_Conductivity.reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        grid_Lag_GW = grid_Lag_GW.reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1]).detach().cpu()
        
        grid_predictions = (grid_predictions * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        grid_Lag_GW = (grid_Lag_GW * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
        grid_Displacement_GW = grid_Displacement_GW * dataset.norm_factors["target_stds"]
        grid_Displacement_S = grid_Displacement_S * dataset.norm_factors["target_stds"]
        grid_Conductivity = grid_Conductivity * dataset.norm_factors["target_stds"]
                                                                                                                                                              
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred_map"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(n_pred)]

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
        if config["forecast_horizon"] is None:
            _, grid_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                    np.datetime64(config["start_date_pred_map"]),
                                                                                    config["n_pred_map"],
                                                                                    iter_pred = iter_pred,
                                                                                    get_displacement_terms = False,
                                                                                    Z_grid = Z_grid)
            n_pred = config["n_pred_map"]
            
        else:
            
            grid_predictions_list = []
            date = np.datetime64(config["start_date_pred_ts"])
            
            for i in range(config["n_pred_map"]):
                _, grid_predictions = plot_ST_MultiPoint.compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                                        date,
                                                                                        config["forecast_horizon"],
                                                                                        iter_pred = iter_pred,
                                                                                        get_displacement_terms = False,
                                                                                        Z_grid = Z_grid)
                grid_predictions_list.append(grid_predictions)
                date = date + np.timedelta64(config["forecast_horizon"], config["frequency"])
                
            grid_predictions = torch.cat(grid_predictions_list, dim = 0)
            n_pred = config["forecast_horizon"] * config["n_pred_map"]
            
            
        grid_predictions = grid_predictions.detach().cpu().reshape(n_pred,config["lat_lon_npoints"][0],config["lat_lon_npoints"][1])
        
        grid_predictions = (grid_predictions * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
                                                                                                                                                            
        start_date_idx = dataset.dates.get_loc(np.datetime64(config["start_date_pred_map"]))
        date_seq = [dataset.dates[start_date_idx+i] for i in range(n_pred)]

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
        
