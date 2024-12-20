from operator import itemgetter

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import fiona

import matplotlib
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
import xarray

from geocube.api.core import make_geocube
from shapely.geometry import box

import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler

from tqdm import tqdm
import time
import wandb
from time import sleep

from models.load_models_2d import Discrete2DNN
from dataloaders.load_2d_meteo_wtd import DiscreteDataset


print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
print(torch.cuda.memory_summary(device=None, abbreviated=False))

json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/discrete_2D_wtd/test_2D.json"
dict_files = {}
with open(json_file) as f:
    dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

timesteps = dict_files["timesteps"]

ds = DiscreteDataset(dict_files)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

print(f"Length of the dataset: {ds.__len__()}")

x,y,z = ds[-1]
print(f"Sizes: {x.shape} - {y.shape} - {z.shape}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


model = Discrete2DNN(timesteps).to(device)

print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

# TEST DATALOADER

batch_size = dict_files["batch_size"]
max_epochs = dict_files["epochs"]

test_split_p = dict_files["test_split_p"]
train_split_p = 1 - test_split_p

max_ds_elems = ds.__len__()
if not dict_files["all_dataset"]:
    max_ds_elems = dict_files["max_ds_elems"]

train_idx = int(max_ds_elems*train_split_p)
test_idx = int(max_ds_elems*test_split_p)

print(f"Traing size: {train_idx}, Test size: {test_idx}")

train_idxs, test_idxs = np.arange(train_idx), np.arange(train_idx, train_idx + test_idx)

train_sampler = SequentialSampler(train_idxs)
test_sampler = SequentialSampler(test_idxs)

train_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=test_sampler)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)


def loss_masked(y_hat,y):
    predict = torch.unsqueeze(y[:,0,:,:,:], dim=1).to(device)
    target = y_hat.to(device)
    mask = y[:,1,:,:,:].bool().to(device)
    out = (torch.sum( (predict - target) * mask) ** 2.0 ) / torch.sum(mask)
    return out


wandb.init(
    entity="gsartor-unito",
    project=dict_files["experiment_name"],
    dir =dict_files["wandb_dir"],
    config=dict_files,
    mode="offline",
)

# Magic
wandb.watch(model, log_freq=100)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
print(torch.cuda.memory_summary(device=None, abbreviated=False))

model.train()


for i in range(max_epochs):
    start_time = time.time()

    print(f"############### Training epoch {i} ###############")
    with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")

                X = (init_wtd.to(device), weather.to(device))
                # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                Y = model(X)
                # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                loss = loss_masked(Y,pred_wtds)
                print(f"Train loss: {loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = {
                    "train_loss" : loss
                }
                wandb.log(metrics)
    

    
    end_time = time.time()
    exec_time = end_time-start_time

    wandb.log({"tr_epoch_exec_t" : exec_time})

    torch.save(model.state_dict(), "model_test.pt")

    print(f"############### Test epoch {i} ###############")

    start_time = time.time()

    with tqdm(test_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")

                X = (init_wtd.to(device), weather.to(device))
                # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                Y = model(X)
                # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                loss = loss_masked(Y,pred_wtds)
                print(f"Test loss: {loss}")

                metrics = {
                    "test_loss" : loss
                }

                wandb.log(metrics)
        
    end_time = time.time()
    exec_time = end_time-start_time
    wandb.log({"test_epoch_exec_t" : exec_time})

wandb.finish()

print(f"Execution time: {end_time-start_time}s")



