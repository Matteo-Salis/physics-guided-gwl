# %% [markdown]
# # Libraries

# %%
import os

# %%
from operator import itemgetter
from tqdm import tqdm
import time
from datetime import datetime
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray
import fiona

#import matplotlib
import matplotlib.pyplot as plt

#from rasterio.enums import Resampling

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
from torch.autograd import Variable

import wandb

import torchview
from torchview import draw_graph

from utils.plot import *
import importlib

# %%
from models import models_2D
from dataloaders import dataset_sparse
from subprocess import Popen

import matplotlib.animation as animation

# %% [markdown]
# # Load dictionary

# %%
json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/VideoCond/SparseData_Transformer_0.json" #config_files_1d/lstm_att_1.json
config = {}
with open(json_file) as f:
    config = json.load(f)
    print(f"Read data.json: {config}")

# %%
saving_path = "/leonardo_scratch/fast/IscrC_DL4EO/results/results_SparseData/plots"
saving_path_maps = f"{saving_path}/maps"
saving_path_ts = f"{saving_path}/time_series"

# %%
ds = dataset_sparse.Dataset_Sparse(config)

# %%
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device

# %%
model = models_2D.SparseData_Transformer(
                weather_CHW_dim = config["weather_CHW_dim"],
                target_dim = config["target_dim"],
                spatial_embedding_dim = config["spatial_embedding_dim"],
                spatial_heads = config["spatial_heads"],
                fusion_embedding_dim = config["fusion_embedding_dim"],
                st_heads = config["st_heads"],
                st_mha_blocks = config["st_mha_blocks"],
                densification_dropout = config["densification_dropout"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"],
                activation= config["activation"]).to(device)

# %%
model_path = "/leonardo_scratch/fast/IscrC_DL4EO/results/results_SparseData/models/model_SparseData_Transformer_20250618_102345.pt"
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
model.eval()


print("Computing predictions...")
# %%
date = "2020-01-05"
twindow = 8
lat_points = 35
lon_points = 45
z_grid = grid_generation(ds, lat_points,lon_points)
Y_test, Y_hat_test = compute_predictions(start_date = np.datetime64(date),
                                        twindow = twindow,
                                        model = model,
                                        device = device,
                                        dataset = ds,
                                        Z_grid = z_grid,
                                        eval = True)

print("Predictions done!")
Y_hat_test_grid = Y_hat_test.reshape(twindow,lat_points,lon_points)
coords = z_grid.reshape(lat_points,lon_points,3)
# Denorm
Y_hat_test_grid = (Y_hat_test_grid * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]
dtm_denorm = (ds.dtm_roi * ds.norm_factors["dtm_std"]) + ds.norm_factors["dtm_mean"]

lat_denorm = (coords[:,0,0] * ds.norm_factors["lat_std"]) + ds.norm_factors["lat_mean"]
lon_denorm = (coords[0,:,1] * ds.norm_factors["lon_std"]) + ds.norm_factors["lon_mean"]

Y_hat_xr = xarray.DataArray(data = Y_hat_test_grid,
                                coords = dict(
                                            lat=("lat", lat_denorm),
                                            lon=("lon", lon_denorm),
                                            time=pd.date_range(np.datetime64(date) + np.timedelta64(1, config["frequency"]),
                                                            np.datetime64(date) + np.timedelta64(twindow, config["frequency"]),
                                                            freq = config["frequency"])),
                                dims = ["time","lat", "lon"]
                                )

dtm_denorm_resampled = dtm_denorm.rio.reproject_match(Y_hat_xr[0].rio.set_spatial_dims(y_dim = "lat", x_dim = "lon").rio.write_crs("epsg:4326"),
                                               resampling = Resampling.average)

WTD_hat_test_grid = dtm_denorm_resampled.values - Y_hat_xr

print("Generating GIF...")
generate_gif_h_wtd(start_date = date, twindow = twindow,
                       sample_h = Y_hat_xr,
                       sample_wtd = WTD_hat_test_grid,
                       freq = "W",
                       save_dir = f'{saving_path_maps}/map_animation_{date}',
                       print_plot = False)
print("GIF saved!")

print("Generating Time Series...")

start_date = '2016-01-03'
n_periods = 16
twindow = 24
start_dates = pd.date_range(start_date, periods = n_periods, freq = f"{twindow}W")

Y_test_list = []
Y_hat_test_list = []
   
for date_idx in range(len(start_dates)):
    Y_test_window, Y_hat_test_window = compute_predictions(
                                        start_date = np.datetime64(start_dates[date_idx]),
                                        twindow = twindow,
                                        model = model,
                                        device = device,
                                        dataset = ds,
                                        eval = True)
    
    
    Y_test_list.append(Y_test_window)
    Y_hat_test_list.append(Y_hat_test_window)
    

Y_test = torch.cat(Y_test_list, dim = 0)
Y_hat_test = torch.cat(Y_hat_test_list, dim = 0)
    
    
Y_hat_test_ds = build_ds_from_pred(Y_hat_test,
                                start_dates[0] + np.timedelta64(1, ds.config["frequency"]),
                                            twindow * n_periods, ds.config["frequency"], ds.sensor_id_list)
Y_test_ds = build_ds_from_pred(Y_test,
                                start_dates[0] + np.timedelta64(1, ds.config["frequency"]),
                                twindow * n_periods, ds.config["frequency"], ds.sensor_id_list)

Y_hat_test_ds = (Y_hat_test_ds * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]
Y_test_ds = (Y_test_ds * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]
            
for sensor in ds.sensor_id_list:
    
    
    municipality = ds.wtd_names["munic"].loc[ds.wtd_names["sensor_id"] == sensor].values[0]
    
    plot_time_series(Y_hat_test_ds[sensor],
                Y_test_ds[sensor],
                title = f"{sensor} - {municipality}",
                save_dir = f'{saving_path_ts}/{municipality}_{sensor}',
                print_plot = False)
    
    
print("Time Series saved!")
