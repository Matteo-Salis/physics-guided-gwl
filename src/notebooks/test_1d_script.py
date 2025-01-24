# %% [markdown]
# # Libraries

# %%
from operator import itemgetter
from tqdm import tqdm
import time
from datetime import datetime
import json
from functools import partial

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

from models.load_models_1d import *
from dataloaders.load_1d_meteo_wtd import ContinuousDataset
from loss.losses_1d import *
from utils.feedforward import *
from plot.prediction_plot_1d import *

# %% [markdown]

# # Load dictionary

# %%
json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D_PINNS_ccnn_att.json"
dict_files = {}
with open(json_file) as f:
    dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

# %%
wandb.init(
    entity="gsartor-unito",
    project=dict_files["experiment_name"],
    dir =dict_files["wandb_dir"],
    config=dict_files,
    mode="offline",
    name=dict_files["run_name"]
)

# %% [markdown]
# # Dataset class


# %%
ds = ContinuousDataset(dict_files)

# %%
print(f"Length of the dataset: {ds.__len__()}")

# %% [markdown]
# # Model 

# %%
device = (
    str(dict_files["cuda_device"])
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Device: ", device)

if dict_files["model"] == "SC_LSTM_idw":
    print("model lstm idw")
    model = SC_LSTM_idw(timestep = dict_files["timesteps"],
                 cb_fc_layer = dict_files["cb_fc_layer"],
                 cb_fc_neurons = dict_files["cb_fc_neurons"],
                 conv_filters = dict_files["conv_filters"],
                 lstm_layer = dict_files["lstm_layer"],
                 lstm_input_units = dict_files["lstm_input_units"],
                 lstm_units = dict_files["lstm_units"]
                 ).to(device)

elif dict_files["model"] == "SC_LSTM_att":
    print("model lstm att")
    model = SC_LSTM_att(timestep = dict_files["timesteps"],
                 cb_emb_dim = dict_files["cb_emb_dim"],
                 cb_att_h = dict_files["cb_att_h"],
                 cb_fc_layer = dict_files["cb_fc_layer"],
                 cb_fc_neurons = dict_files["cb_fc_neurons"],
                 conv_filters = dict_files["conv_filters"],
                 lstm_layer = dict_files["lstm_layer"],
                 lstm_input_units = dict_files["lstm_input_units"],
                 lstm_units = dict_files["lstm_units"]
                 ).to(device)
    
elif dict_files["model"] == "SC_CCNN_att":
    print("model causal cnn att")
    model = SC_CCNN_att(timestep = dict_files["timesteps"],
                 cb_emb_dim = dict_files["cb_emb_dim"],
                 cb_att_h = dict_files["cb_att_h"],
                 cb_fc_layer = dict_files["cb_fc_layer"],
                 cb_fc_neurons = dict_files["cb_fc_neurons"],
                 conv_filters = dict_files["conv_filters"],
                 ccnn_input_filters =  dict_files["ccnn_input_filters"],
                 ccnn_kernel_size =  dict_files["ccnn_kernel_size"],
                 ccnn_n_filters =  dict_files["ccnn_n_filters"],
                 ccnn_n_layers =  dict_files["ccnn_n_layers"],
                 ).to(device)
    
elif dict_files["model"] == "SC_CCNN_idw":
    print("model causal cnn idw")
    model = SC_CCNN_idw(timestep = dict_files["timesteps"],
                 cb_emb_dim = dict_files["cb_emb_dim"],
                 cb_att_h = dict_files["cb_att_h"],
                 cb_fc_layer = dict_files["cb_fc_layer"],
                 cb_fc_neurons = dict_files["cb_fc_neurons"],
                 conv_filters = dict_files["conv_filters"],
                 ccnn_input_filters =  dict_files["ccnn_input_filters"],
                 ccnn_kernel_size =  dict_files["ccnn_kernel_size"],
                 ccnn_n_filters =  dict_files["ccnn_n_filters"],
                 ccnn_n_layers =  dict_files["ccnn_n_layers"],
                 ).to(device)

# Magic
wandb.watch(model, log_freq=100)

# %%
print("Total number of trainable parameters: " ,sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% [markdown]
# # Training

# %%
batch_size = dict_files["batch_size"]
max_epochs = dict_files["epochs"]

max_ds_elems = ds.__len__()
if not dict_files["all_dataset"]:
    max_ds_elems = dict_files["max_ds_elems"]
    
if type(dict_files["test_split_p"]) is str:
    
    train_idx = int(ds.get_iloc_from_date(date_max= np.datetime64(dict_files["test_split_p"])))
    test_idx = int(max_ds_elems - train_idx)
else:
    test_split_p = dict_files["test_split_p"]
    train_split_p = 1 - test_split_p
    
    train_idx = int(max_ds_elems*train_split_p)
    test_idx = int(max_ds_elems*test_split_p)

train_idxs, test_idxs = np.arange(train_idx), np.arange(train_idx,
                                                        train_idx + test_idx)

print(f"Traing size: {train_idx} - {ds.wtd_df.index.get_level_values(0)[train_idxs[-1]]}, Test size: {test_idx} - {ds.wtd_df.index.get_level_values(0)[test_idxs[-1]]}")

# Sampler 
if dict_files["random_sampler"] is True:
    train_sampler = RandomSampler(train_idxs)
else:
    train_sampler = SequentialSampler(train_idxs)
    
test_sampler = SequentialSampler(test_idxs)

# DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=test_sampler)


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=dict_files["lr"])

# %%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = 'model_{}'.format(timestamp) 

# Loss

if dict_files["loss"] == "data":
    forward_and_loss = feedforward_dloss
    
elif dict_files["loss"] == "data+pde":
    
    ## DA CAMBIARE VANNO NEL MODELLO!!!
    g = torch.nn.Parameter(torch.FloatTensor([dict_files["pde_g"][0]]),
                                            requires_grad=dict_files["pde_g"][1]).to(device)
    
    k_lat = torch.nn.Parameter(torch.FloatTensor([dict_files["pde_k_lat"][0]]),
                                            requires_grad=dict_files["pde_k_lat"][1]).to(device)
    
    k_lon = torch.nn.Parameter(torch.FloatTensor([dict_files["pde_k_lon"][0]]),
                                            requires_grad=dict_files["pde_k_lon"][1]).to(device)
    
    S_y = torch.nn.Parameter(torch.FloatTensor([dict_files["pde_Sy"][0]]),
                                            requires_grad=dict_files["pde_Sy"][1]).to(device)
    
    lon_cpoints = dict_files["lon_cpoints"]
    
    coeff_loss_data = dict_files["coeff_loss_data"]
    coeff_loss_pde = dict_files["coeff_loss_pde"]
    
    forward_and_loss = partial(feedforward_dloss_pdeloss, 
                                        g = g,
                                        k_lat = k_lat,
                                        k_lon = k_lon,
                                        S_y = S_y,
                                        lon_cpoints = lon_cpoints,
                                        ds = ds,
                                        device = device,
                                        coeff_loss_data = coeff_loss_data,
                                        coeff_loss_pde = coeff_loss_pde)


weather_coords = ds.get_weather_coords()
weather_dtm = ds.get_weather_dtm()
weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
print(torch.cuda.memory_summary(device=device, abbreviated=False))

for i in range(max_epochs):
    
    model.train(True)
    start_time = time.time()
    print(f"############### Training epoch {i} ###############")
    
    with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")
                
                x = x.to(device)
                x_mask = x_mask.to(device)
                z = z.to(device)
                weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                w = [w_values.to(device), weather_coords_batch.to(device)]
                y = y.to(device)
                y_mask = y_mask.to(device)
                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                
                optimizer.zero_grad()
                
                #y_hat = model(x, z, w, x_mask)
                loss_dict = forward_and_loss(model = model,
                                             input = (x, z, w, x_mask),
                                             groundtruth = (y, y_mask),
                                             loss_prefix_name = "Train")
                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                # loss = masked_mse(y_hat,
                #                   y,
                #                   y_mask)
                print({key:loss_dict[key].item() for key in list(loss_dict)})
                
                loss_dict[list(loss_dict)[-1]].backward()
                optimizer.step()
                
                # metrics = {
                #     "train_loss" : loss
                # }
                wandb.log({key:loss_dict[key].item() for key in list(loss_dict)})              
                
    end_time = time.time()
    exec_time = end_time-start_time

    wandb.log({"tr_epoch_exec_t" : exec_time})
    
    model_dir = dict_files["save_model_dir"]
    torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt") 

    print(f"############### Test epoch {i} ###############")
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    start_time = time.time()
    # Disable gradient computation and reduce memory consumption.
    #with torch.no_grad():
    with tqdm(test_loader, unit="batch") as tepoch:
                for batch_idx, (x, z, w_values, y, x_mask, y_mask) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")

                    x = x.to(device)
                    x_mask = x_mask.to(device)
                    z = z.to(device)
                    weather_coords_batch = weather_coords.unsqueeze(0).expand(w_values.shape[0], -1, -1, -1)
                    w = [w_values.to(device), weather_coords_batch.to(device)]
                    y = y.to(device)
                    y_mask = y_mask.to(device)
                    #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    #y_hat = model(x, z, w, x_mask)
                    loss_dict = forward_and_loss(model = model,
                                             input = (x, z, w, x_mask),
                                             groundtruth = (y, y_mask),
                                             loss_prefix_name = "Test")
                    #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    # loss = masked_mse(y_hat,
                    #               y,
                    #               y_mask)
                    print({key:loss_dict[key].item() for key in list(loss_dict)})

                    # metrics = {
                    #     "test_loss" : loss
                    # }

                    wandb.log({key:loss_dict[key].item() for key in list(loss_dict)})
        
    end_time = time.time()
    exec_time = end_time-start_time
    wandb.log({"test_epoch_exec_t" : exec_time})

    # Plots
    for date in dict_files["plot_dates"]:
                # Time Series  
                wandb.log({f"pred_series_{date} - R":wandb.Image(plot_one_series(ds = ds,
                                                             date_t0 = np.datetime64(date),
                                                             sensor = 6,
                                                             model = model,
                                                             device = device,
                                                             print_plot = False))})
    
                wandb.log({f"pred_series_{date} - V":wandb.Image(plot_one_series(ds = ds,
                                                             date_t0 = np.datetime64(date),
                                                             sensor = 15,
                                                             model = model,
                                                             device = device,
                                                             print_plot = False))})
                
                # Maps
                sample_h, sample_wtd, dtm_denorm_downsampled = predict_map_points(ds, lon_point = 40, 
                            sample_date = date,
                            model = model, device = device)
                
                for tstep in dict_files["plot_tstep_map"]:
    
                        wandb.log({f"pred_map_{date}-t{tstep}":wandb.Image(plot_one_map(sample_h, sample_wtd, dtm_denorm_downsampled, 
                                    date, pred_timestep = tstep,
                                    save_dir = None, 
                                    print_plot = False))})

model_graph = draw_graph(model, input_data=(x, z, w, x_mask), device=device)
model_graph.visual_graph.render(format='png', filename = model_name, directory= f"{model_dir}/")
model_arch = wandb.Image(f"{model_dir}/{model_name}.png", caption="model's architecture")
wandb.log({"model_arch": model_arch})
                
wandb.finish()

print(f"Execution time: {end_time-start_time}s")

# %%



