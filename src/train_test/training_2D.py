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


def train_dl_model(epoch, dataset, model, train_loader, loss_fn, optimizer, model_dir, model_name,
                      start_dates_plot, twindow_plot, sensors_to_plot, timesteps_to_look,
                      device = "cuda", plot_arch = True): #, l2_alpha = 0.0005
    
    with tqdm(train_loader, unit="batch") as tepoch:
                    
                    for batch_idx, (X, Z, W, Y, X_mask, Y_mask) in enumerate(tepoch):
                        tepoch.set_description(f"Epoch {epoch}")
                        
                        X = X.to(device)
                        X_mask = X_mask.to(device)
                        Z = Z.to(device)
                        W = [W[0].to(device), W[1].to(device)]
                        Y = Y.to(device)
                        Y_mask = Y_mask.to(device)
                        #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        
                        optimizer.zero_grad()
                        
                        Y_hat = model(X, Z, W, X_mask)
                        
                        #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                        loss = loss_fn(Y_hat,
                                          Y,
                                          Y_mask)
                        
                        #loss += l2_alpha * loss_l2_regularization(model)
                        
                        print("Training_data_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Training_data_loss":loss.item()})   
                        
                    # Plots
                    model.eval()
                    with torch.no_grad():
                        plot_maps_and_time_series(dataset, model, device,
                              start_dates_plot, twindow_plot,
                              sensors_to_plot, 
                              timesteps_to_look)
                        
                        if epoch == 0 and plot_arch is True:
                            print("Saving plot of the model's architecture...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (dataset[0][0].unsqueeze(0),
                                                                                    dataset[0][1].unsqueeze(0),
                                                                                    [dataset[0][2][0].unsqueeze(0),
                                                                                    dataset[0][2][1].unsqueeze(0)],
                                                                                    dataset[0][-2].unsqueeze(0)),
                                                                                    device = device)})
                            