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
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from scipy import signal

from tqdm import tqdm
import time
from datetime import datetime
import wandb
from time import sleep

from models.load_models_2d import *
from dataloaders.load_2d_meteo_wtd import DiscreteDataset


print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
print(torch.cuda.memory_summary(device=None, abbreviated=False))

json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/discrete_2D_wtd/test_2D_blocks_normalized.json"
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

model = None

if dict_files["model"] == "Discrete2DMidConcatNN":
    model = Discrete2DMidConcatNN(timesteps).to(device)
elif dict_files["model"] == "Discrete2DNN":
    model = Discrete2DNN(timesteps).to(device)
else:
     raise Exception("Model name unknown.")

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

train_sampler = RandomSampler(train_idxs)
test_sampler = RandomSampler(test_idxs)

train_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=test_sampler)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

def ConvLon(x):
    print(x.shape)
    conv = torch.Tensor([[-1,2,-1],[-1,2,-1],[-1,2,-1]] )
    out = torch.Tensor(x.shape)

    for b in range(x.shape[0]):
        for t in range(x.shape[2]):
            out[b,0,t,:,:] = torch.from_numpy(signal.convolve2d(x[b,0,t,:,:].cpu().detach().numpy(), conv, mode='same'))
    return out

def ConvLat(x):
    conv = torch.Tensor([[-1,-1,-1],[2,2,2],[-1,-1,-1]] )
    out = torch.Tensor(x.shape)

    for b in range(x.shape[0]):
        for t in range(x.shape[2]):
            out[b,0,t,:,:] = torch.from_numpy(signal.convolve2d(x[b,0,t,:,:].cpu().detach().numpy(), conv, mode='same'))

    # out = signal.convolve2d(x, conv, mode='same')
    return out

def pde_grad_loss(y_hat):
    # already normalized and computed on height
    print(y_hat.shape)

    y_hat.to(device)

    lat_grad = ConvLat(y_hat).to(device)
    lon_grad = ConvLon(y_hat).to(device)

    # y_hat_t_1 - y_hat_t = lat_grad_t + lon_grad_t
    loss = torch.sum(y_hat[:,0,1:-1,:,:] - y_hat[:,0,0:-2,:,:] + lat_grad[:,0,0:-2,:,:] + lon_grad[:,0,0:-2,:,:] )

    return torch.abs(loss)

def pde_grad_loss_darcy(y_hat):
    # normalized and working on height
    # print(y_hat.shape)

    y_hat.to(device)

    k_x = 1  # TODO: to esimate
    k_y = 1  # TODO: to esimate

    lat_grad = ConvLat(y_hat).to(device)
    lon_grad = ConvLon(y_hat).to(device)

    dh_t = y_hat[:,0,1:-1,:,:] - y_hat[:,0,0:-2,:,:]
    dh_x = lon_grad[:,0,0:-2,:,:]
    d2_x = ConvLon(torch.unsqueeze(dh_x, dim=1)).to(device)
    dh_y = lat_grad[:,0,0:-2,:,:]
    d2_y = ConvLat(torch.unsqueeze(dh_y, dim=1)).to(device)

    G = 0 # TODO: to esimate

    loss = dh_t - (d2_x * (-k_x * dh_x * y_hat[:,0,0:-2,:,:]) ) - (d2_y * (-k_y * dh_y * y_hat[:,0,0:-2,:,:]))
    loss = torch.sum(loss ** 2) / torch.numel(loss)

    return loss

def pde_grad_loss_wtd(y_hat,dtm,wtd_mean,wtd_std):
    # TODO: normalized or de-normalized?
    print(y_hat.shape)

    predict = (y_hat * wtd_std) + wtd_mean # denormalized
    predict[:,0,:,:,:] = - predict[:,0,:,:,:] + dtm[0,:,:]
    predict.to(device)

    lat_grad = ConvLat(predict).to(device)
    lon_grad = ConvLon(predict).to(device)

    # y_hat_t_1 - y_hat_t = lat_grad_t + lon_grad_t
    loss = torch.sum(predict[:,0,1:-1,:,:] - predict[:,0,0:-2,:,:] + lat_grad[:,0,0:-2,:,:] + lon_grad[:,0,0:-2,:,:] )

    return torch.abs(loss)

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

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = 'model_{}.pt'.format(timestamp)    
c1_loss = dict_files["c1_loss"]
c2_loss = dict_files["c2_loss"]

dtm = torch.from_numpy(ds.dtm_roi_downsampled.values).to(device)
wtd_mean = ds.wtd_numpy_mean
wtd_std = ds.wtd_numpy_std

Y = None

for i in range(max_epochs):
    model.train()

    start_time = time.time()

    print(f"############### Training epoch {i} ###############")
    with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")

                X = (init_wtd.to(device), weather.to(device))
                # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                Y = model(X)
                # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                loss_mask = loss_masked(Y,pred_wtds)
                loss_pde = pde_grad_loss_darcy(Y)
                loss = c1_loss * loss_mask + c2_loss * loss_pde
                print(f"Train loss: {loss}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics = {
                    "train_loss_mask" : loss_mask,
                    "train_loss_pde" : loss_pde,
                    "train_loss" : loss
                }
                wandb.log(metrics)
    

    
    end_time = time.time()
    exec_time = end_time-start_time

    wandb.log({"tr_epoch_exec_t" : exec_time})

    torch.save(model.state_dict(), f"{dict_files['save_model_dir']}/{model_name}")

    # plots on wandb
    with torch.no_grad():
        predict = (Y.cpu() * wtd_std) + wtd_mean
        plt.figure(figsize = (10,10))
        plt.imshow(predict[0,0,0,:,:])
        plt.colorbar()
        plt.savefig(f"predict_a{i}.png", bbox_inches = 'tight')
        wandb.log({
            "train_prediction" :  wandb.Image(f"predict_a{i}.png", caption="prediction A on training")
        })

    with torch.no_grad():
        predict = (Y.cpu() * wtd_std) + wtd_mean
        plt.figure(figsize = (10,10))
        plt.imshow(predict[0,0,100,:,:])
        plt.colorbar()
        plt.savefig(f"predict_b{i}.png", bbox_inches = 'tight')
        wandb.log({
            "train_prediction" :  wandb.Image(f"predict_b{i}.png", caption="prediction B on training")
        })

    print(f"############### Test epoch {i} ###############")

    model.eval()
    start_time = time.time()

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
                for batch_idx, (init_wtd, weather, pred_wtds) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")

                    X = (init_wtd.to(device), weather.to(device))
                    # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    Y = model(X)
                    # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    loss_mask = loss_masked(Y,pred_wtds)
                    loss_pde = loss_pde = pde_grad_loss_darcy(Y)
                    loss = c1_loss * loss_mask + c2_loss * loss_pde
                    print(f"Test loss: {loss}")

                    metrics = {
                        "test_loss_mask" : loss_mask,
                        "test_loss_pde" : loss_pde,
                        "test_loss" : loss
                    }

                    wandb.log(metrics)
        
    end_time = time.time()
    exec_time = end_time-start_time
    wandb.log({"test_epoch_exec_t" : exec_time})

    # plots on wandb
    with torch.no_grad():
        predict = (Y.cpu() * wtd_std) + wtd_mean
        plt.figure(figsize = (10,10))
        plt.imshow(predict[0,0,0,:,:])
        plt.colorbar()
        plt.savefig(f"predict_test_a{i}.png", bbox_inches = 'tight')
        wandb.log({
            "test_prediction" :  wandb.Image(f"predict_test_a{i}.png", caption="prediction A on test")
        })

    with torch.no_grad():
        predict = (Y.cpu() * wtd_std) + wtd_mean
        plt.figure(figsize = (10,10))
        plt.imshow(predict[0,0,100,:,:])
        plt.colorbar()
        plt.savefig(f"predict_test_b{i}.png", bbox_inches = 'tight')
        wandb.log({
            "test_prediction" :  wandb.Image(f"predict_test_b{i}.png", caption="prediction B on test")
        })

wandb.finish()

print(f"Execution time: {end_time-start_time}s")



