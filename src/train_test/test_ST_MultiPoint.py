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

from utils.plot_ST_MultiPoint import *
from loss.losses_ST_MultiPoint import *
    
    
def pure_dl_tester(epoch, dataset, model, test_loader, loss_fn,
                    start_dates_plot, n_pred_plot, sensors_to_plot, t_step_to_plot, lat_lon_points,
                    #timesteps_to_look,
                    device = "cuda"):
    
    with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                            
                            for batch_idx, (X, W, Z, Y) in enumerate(tepoch):
                                tepoch.set_description(f"Epoch {epoch}")

                                X = [X[0].to(device),
                                      X[1].to(device),
                                      X[2].to(device)]
                        
                                W = [W[0].to(device),
                                    W[1].to(device)]
                                
                                Z = Z.to(device)
                                
                                Y = [Y[0].to(device),
                                    Y[1].to(device)]

                                Y_hat = model(X, W, Z, mc_dropout = False)
                                
                                loss = loss_fn(Y_hat,
                                          Y[0],
                                          Y[1])
                                
                                print("Test_data_loss: ", loss.item())
                                wandb.log({"Test_data_loss":loss.item()})
                            
                            if (epoch+1) % 25 == 0:
                              wandb_time_series(dataset, model, device,
                                start_dates_plot, n_pred_plot,
                                sensors_to_plot,
                                eval_mode=False)
                              
                              wandb_video(dataset, model, device,
                                      start_dates_plot, n_pred_plot,
                                      t_step_to_plot,
                                      lat_points = lat_lon_points[0],
                                      lon_points= lat_lon_points[1],
                                      eval_mode = False)
                            
                            if (epoch+1) % 50 == 0:
                            
                              print("Computing iterated predictions...")
                              
                              wandb_time_series(dataset, model, device,
                                start_dates_plot, n_pred_plot,
                                sensors_to_plot,
                                eval_mode = True)
                              
                              wandb_video(dataset, model, device,
                                      start_dates_plot, n_pred_plot,
                                      t_step_to_plot,
                                      lat_points = lat_lon_points[0],
                                      lon_points= lat_lon_points[1],
                                      eval_mode = True)
                            
                            
# def test_pinns_model(epoch, dataset, model, test_loader, loss_fn, loss_physics_fn,
#                     start_dates_plot, twindow_plot, sensors_to_plot, timesteps_to_look,
#                     device = "cuda"):
    
#     with torch.no_grad():
#                 with tqdm(test_loader, unit="batch") as tepoch:
                            
#                             for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
#                                 tepoch.set_description(f"Epoch {epoch}")

#                                 X = X[:,0,:,:].to(device)
#                                 X_mask = X_mask[:,0,:].to(device)
#                                 Z = Z.to(device)
#                                 W = [W[0].to(device), W[1].to(device)]
#                                 Y = Y.to(device)
#                                 Y_mask = Y_mask.to(device)
#                                 #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 Y_hat, K_lat_lon = model(X, Z, W, X_mask,
#                                                          mc_dropout = False,
#                                                          K_out = True)
                                
#                                 #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

#                                 loss_data = loss_fn(Y_hat,
#                                           Y,
#                                           Y_mask)
                                
#                                 loss_physics = loss_physics_fn(Y_hat,
#                                                               K_lat = K_lat_lon[:,0,:,:].unsqueeze(1),
#                                                               K_lon = K_lat_lon[:,1,:,:].unsqueeze(1))
                                
#                                 print(f"Test_data_loss: {loss_data.item()} --- Test_physics_loss: {loss_physics.item()}")
#                                 wandb.log({"Test_data_loss":loss_data.item()})
#                                 wandb.log({"Test_physics_loss":loss_physics.item()})
                            
#                             plot_maps_and_time_series(dataset, model, device,
#                               start_dates_plot, twindow_plot,
#                               sensors_to_plot, 
#                               timesteps_to_look,
#                               eval_mode=True)
                            
#                             K_lat_lon = build_xarray(K_lat_lon[0].detach().cpu(), dataset, variable = "K_lat_lon")
#                             wandb.log({"K_maps_test":wandb.Image(plot_K_lat_lon_maps(K_lat_lon))})