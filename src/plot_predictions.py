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
from models.models_1d import *
from dataloaders.dataset_1d import Dataset_1D
from dataloaders import dataset_2D
from dataloaders.dataset_2d import *
from subprocess import Popen

from models import models_2D

import matplotlib.animation as animation

# %% [markdown]
# # Load dictionary

# %%
json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/VideoCond/ConvLSTM_VideoCond_focal_pinns_0.json" #config_files_1d/lstm_att_1.json
dict_files = {}
with open(json_file) as f:
    dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

# %%
saving_path = "/leonardo_scratch/fast/IscrC_DL4EO/results/results_2D/plots"
saving_path_maps = f"{saving_path}/maps"
saving_path_ts = f"{saving_path}/time_series"

sample_date = "2019-01-01"
twindow = 365

# %%
ds_2D = dataset_2D.Dataset_2D_VideoCond(dict_files)

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device

# %%
model = models_2D.VideoCB_ConvLSTM(
                weather_CHW_dim = dict_files["weather_CHW_dim"],
                cb_emb_dim = dict_files["cb_emb_dim"],
                cb_heads = dict_files["cb_heads"],
                channels_cb = dict_files["channels_cb"],
                channels_wb = dict_files["channels_wb"],
                convlstm_IO_units = dict_files["convlstm_IO_units"],
                convlstm_hidden_units = dict_files["convlstm_hidden_units"],
                convlstm_kernel = dict_files["convlstm_kernel"],
                convlstm_nlayer = dict_files["convlstm_nlayer"],
                densification_dropout = dict_files["densification_dropout"],
                upsampling_dim = dict_files["upsampling_dim"],
                layernorm_affine = dict_files["layernorm_affine"],
                spatial_dropout = dict_files["spatial_dropout"]).to(device)

# %%
model_path = "/leonardo_scratch/fast/IscrC_DL4EO/results/results_2D/models/model_VideoCB_ConvLSTM_20250522_230917.pt"
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
model.eval()


print("Computing predictions...")
# %%
Y_test, Y_hat_test = test_data_prediction(start_date = np.datetime64(sample_date),
                                            twindow = twindow,
                                            model = model,
                                            device = device,
                                            dataset = ds_2D,
                                            eval = True)

print("Predictions done!")
# %%
Y_hat_test_xr_denorm = build_xarray(norm_data = Y_hat_test,
                         dataset = ds_2D,
                         start_date = sample_date,
                         twindow = twindow)
            
WTD_hat_test_xr_denorm = ds_2D.target_rasterized_dtm.values - Y_hat_test_xr_denorm
            
Y_test_xr_denorm = build_xarray(norm_data = Y_test,
                dataset = ds_2D,
                start_date = sample_date,
                twindow = twindow)

print("Generating GIF...")
generate_gif_h_wtd(start_date = sample_date, twindow = twindow,
                       sample_h = Y_hat_test_xr_denorm,
                       sample_wtd = WTD_hat_test_xr_denorm,
                       save_dir = f'{saving_path_maps}/map_animation_{sample_date}',
                       print_plot = False)

print("GIF saved!")
# %%
ds_2D.sensor_id_list


print("Generating Time Series...")
# %%
for sensor_id in ds_2D.sensor_id_list:
    municipality, lat, lon = find_munic_lat_lon_sensor(ds_2D, sensor_id)
    sensor_pred_ds = find_sensor_pred_in_xr(Y_test_xr_denorm, Y_hat_test_xr_denorm,
                                                            lat = lat,
                                                            lon = lon,
                                                            )
    
    plot_sensor_ts(sensor_pred_ds,
                    title = f"{sensor_id} - {municipality} - from {sample_date}",
                    save_dir = f'{saving_path_ts}/{sample_date}_{municipality}_{sensor_id}',
                    print_plot = False)

print("Time Series saved!")

