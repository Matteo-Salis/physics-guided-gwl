import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import time
from tqdm import tqdm
import wandb

from torchview import draw_graph

from utils.plot import *
from loss.losses_2D import *
    
    
def test_dl_model(epoch, dataset, model, test_loader, loss_fn,
                    start_dates_plot, twindow_plot, sensors_to_plot, timesteps_to_look,
                    device = "cuda"):
    
    with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            
                            for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {epoch}")

                                X = X[:,0,:,:].to(device)
                                X_mask = X_mask[:,0,:].to(device)
                                Z = Z.to(device)
                                W = [W[0].to(device), W[1].to(device)]
                                Y = Y.to(device)
                                Y_mask = Y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                Y_hat = model(X, Z, W, X_mask, mc_dropout = False)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss = loss_fn(Y_hat,
                                          Y,
                                          Y_mask)
                                
                                print("Test_data_loss: ", loss.item())
                                wandb.log({"Test_data_loss":loss.item()})
                            
                            plot_maps_and_time_series(dataset, model, device,
                              start_dates_plot, twindow_plot,
                              sensors_to_plot, 
                              timesteps_to_look,
                              eval_mode=True)
                            
                            
def test_pinns_model(epoch, dataset, model, test_loader, loss_fn, loss_physics_fn,
                    start_dates_plot, twindow_plot, sensors_to_plot, timesteps_to_look,
                    device = "cuda"):
    
    with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            
                            for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {epoch}")

                                X = X[:,0,:,:].to(device)
                                X_mask = X_mask[:,0,:].to(device)
                                Z = Z.to(device)
                                W = [W[0].to(device), W[1].to(device)]
                                Y = Y.to(device)
                                Y_mask = Y_mask.to(device)
                                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                Y_hat = model(X, Z, W, X_mask, mc_dropout = False)
                                
                                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                                loss_data = loss_fn(Y_hat,
                                          Y,
                                          Y_mask)
                                
                                loss_physics = loss_physics_fn(Y_hat)
                                
                                print(f"Test_data_loss: {loss_data.item()} --- Test_physics_loss: {loss_physics.item()}")
                                wandb.log({"Test_data_loss":loss_data.item()})
                                wandb.log({"Test_physics_loss":loss_physics.item()})
                            
                            plot_maps_and_time_series(dataset, model, device,
                              start_dates_plot, twindow_plot,
                              sensors_to_plot, 
                              timesteps_to_look,
                              eval_mode=True)
                            