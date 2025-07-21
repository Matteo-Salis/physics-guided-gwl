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

from loss.PytorchPCGrad.pcgrad import PCGrad

from torch import autograd


def train_dl_model_SparseData(epoch, dataset, model, train_loader, loss_fn, optimizer, model_dir, model_name,
                      start_dates_plot, twindow_plot, sensors_to_plot,
                      #timesteps_to_look,
                      teacher_forcing_factor = 1,
                      device = "cuda", plot_arch = True, l2_alpha = 0): #, 
    
    with tqdm(train_loader, unit="batch") as tepoch:
        #with autograd.detect_anomaly():
                    
                    for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {epoch}")
                        
                        teacher_forcing = torch.rand(1).item()
                        if teacher_forcing > teacher_forcing_factor:
                            teacher_forcing = False
                            X = X[:,0,:,:].to(device)
                            X_mask = X_mask[:,0,:].to(device)
                            
                        else:
                            teacher_forcing = True
                            print("Teacher Forcing Mode!", end = " ")
                            X = X.to(device)
                            X_mask = X_mask.to(device)
                            
                        Z = Z.to(device)
                        W = [W[0].to(device), W[1].to(device)]
                        Y = Y.to(device)
                        Y_mask = Y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        if "MoE" in model_name:
                            Y_hat, aux_loss = model(X, Z, W, X_mask, teacher_forcing = teacher_forcing, mc_dropout = True,
                                                        get_aux_loss = True)
                            moe = True
                        else:
                            Y_hat = model(X, Z, W, X_mask, teacher_forcing = teacher_forcing, mc_dropout = True)
                            moe = False
                            
                        loss = loss_fn(Y_hat,
                                          Y,
                                          Y_mask)
                        
                        if l2_alpha > 0:
                            loss += l2_alpha * loss_l2_regularization(model)
                        
                        if moe is True:
                            loss += aux_loss
                            print("Training_data_loss: ", loss.item(),
                              end = " - "
                              )
                            print("aux_moe_loss: ", aux_loss)
                            
                        else:
                             print("Training_data_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Training_data_loss":loss.item()})   
                        
                    # Plots
                    #model.eval()
                    with torch.no_grad():
                        wandb_time_series(dataset, model, device,
                              start_dates_plot, twindow_plot,
                              sensors_to_plot, 
                              #timesteps_to_look,
                              eval_mode = False)
                        
                        if epoch == 0 and plot_arch is True:
                            print("Saving plot of the model's architecture...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (dataset[0][0].unsqueeze(0),
                                                                                    dataset[0][1].unsqueeze(0),
                                                                                    [dataset[0][2][0].unsqueeze(0),
                                                                                    dataset[0][2][1].unsqueeze(0)],
                                                                                    dataset[0][-2].unsqueeze(0),
                                                                                    True),
                                                                                    device = device)})
                            
                            
# def train_pinns_model_SparseData(epoch, dataset, model, train_loader,
#                       loss_fn, loss_physics_fn, losses_coeff, physics_guide_alpha,
#                       optimizer, model_dir, model_name,
#                       start_dates_plot, twindow_plot, sensors_to_plot, timesteps_to_look, teacher_forcing_factor = 1,
#                       device = "cuda", plot_arch = True): #, l2_alpha = 0.0005
    
#     physics_guide_alpha = physics_guide_alpha.to(device)
#     print(f"Physics Guide Alpha: {physics_guide_alpha[epoch]}")
    
#     with tqdm(train_loader, unit="batch") as tepoch:
#         #with autograd.detect_anomaly():
                    
#                     for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
#                         tepoch.set_description(f"Epoch {epoch}")
                        
#                         teacher_forcing = torch.rand(1).item()
#                         if teacher_forcing > teacher_forcing_factor:
#                             teacher_forcing = False
#                             X = X[:,0,:,:].to(device)
#                             X_mask = X_mask[:,0,:].to(device)
                            
#                         else:
#                             teacher_forcing = True
#                             print("Teacher Forcing Mode!", end = " - ")
#                             X = X.to(device)
#                             X_mask = X_mask.to(device)
                            
#                         Z = Z.to(device)
#                         W = [W[0].to(device), W[1].to(device)]
#                         Y = Y.to(device)
#                         Y_mask = Y_mask.to(device)
#                         #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
#                         optimizer.zero_grad()
                        
#                         Y_hat, K_lat_lon = model(X, Z, W, X_mask, teacher_forcing = teacher_forcing,
#                                                  mc_dropout = True, K_out = True)
                        
#                         data_loss = loss_fn(Y_hat,
#                                             Y,
#                                             Y_mask)
                        
#                         wandb.log({"Training_data_loss":data_loss.item()})
                        
#                         physics_guide = torch.rand(1).item()
                        
#                         if physics_guide < physics_guide_alpha[epoch]:
                            
#                             physics_loss = loss_physics_fn(Y_hat,
#                                                             K_lat = K_lat_lon[:,0,:,:].unsqueeze(1),
#                                                             K_lon = K_lat_lon[:,1,:,:].unsqueeze(1))
#                             print(f"Training_data_loss: {data_loss.item()} --- Training_physics_loss: {physics_loss.item()}")
#                             loss = losses_coeff[0]*data_loss + losses_coeff[1]*physics_loss
                            
#                             wandb.log({"Training_physics_loss":physics_loss.item()})
                            
#                         else:
                            
#                             print(f"Training_data_loss: {data_loss.item()}")
#                             loss = data_loss
                            
                        
#                         loss.backward()
#                         optimizer.step()
                        
                          
                        
                        
#                     # Plots
#                     model.eval()
#                     with torch.no_grad():
#                         plot_maps_and_time_series(dataset, model, device,
#                               start_dates_plot, twindow_plot,
#                               sensors_to_plot, 
#                               timesteps_to_look,
#                               eval_mode = True)
                        
#                         if epoch == 0 and plot_arch is True:
#                             print("Saving plot of the model's architecture...")
#                             wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
#                                                                       sample_input = (dataset[0][0].unsqueeze(0),
#                                                                                     dataset[0][1].unsqueeze(0),
#                                                                                     [dataset[0][2][0].unsqueeze(0),
#                                                                                     dataset[0][2][1].unsqueeze(0)],
#                                                                                     dataset[0][-2].unsqueeze(0),
#                                                                                     True),
#                                                                                     device = device)})
                            
#                         K_lat_lon = build_xarray(K_lat_lon[0].detach().cpu(), dataset, variable = "K_lat_lon")
#                         wandb.log({"K_maps_train":wandb.Image(plot_K_lat_lon_maps(K_lat_lon))})
                            