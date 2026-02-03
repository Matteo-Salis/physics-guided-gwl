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
                    model_dir,
                    device = "cuda", plot_displacements = False):
    
    """ training procedure to test pure deep learning models
    used also for physics guided models given physics loss

    Args:
        epoch (_type_): _description_
        dataset (_type_): _description_
        model (_type_): _description_
        test_loader (_type_): _description_
        loss_fn (_type_): _description_
        start_dates_plot (_type_): _description_
        n_pred_plot (_type_): _description_
        sensors_to_plot (_type_): _description_
        t_step_to_plot (_type_): _description_
        lat_lon_points (_type_): _description_
        model_dir (_type_): _description_
        device (str, optional): _description_. Defaults to "cuda".
        plot_displacements (bool, optional): _description_. Defaults to False.
    """
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
                              predict_and_plot_time_series(dataset, model, device,
                                start_dates_plot, n_pred_plot,
                                sensors_to_plot,
                                eval_mode=False,
                                log_wandb=False,
                                save_dir=model_dir,
                                title_ext = f"E{epoch}")
                              
                            
                              if plot_displacements is False:
                                predict_and_plot_video(dataset, model, device,
                                        start_dates_plot, n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = False,
                                        log_wandb=False,
                                        save_dir=model_dir,
                                        title_ext = f"E{epoch}")
                              
                              else:
                                predict_and_plot_video_displacements(dataset, model, device,
                                        start_dates_plot, n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = False,
                                        log_wandb=False,
                                        save_dir=model_dir,
                                        title_ext = f"E{epoch}")
                            
                            # uncomment if you want to plot images
                            # if (epoch+1) % 50 == 0:
                            
                            #   print("Computing iterated predictions...")
                              
                            #   predict_and_plot_time_series(dataset, model, device,
                            #     [start_dates_plot[-1]], n_pred_plot,
                            #     sensors_to_plot,
                            #     eval_mode = True,
                            #     log_wandb=False,
                            #     save_dir=model_dir,
                            #     title_ext = f"E{epoch}")
                              
                            #   if plot_displacements is False:
                            #     predict_and_plot_video(dataset, model, device,
                            #             [start_dates_plot[-1]], n_pred_plot,
                            #             t_step_to_plot,
                            #             lat_points = lat_lon_points[0],
                            #             lon_points= lat_lon_points[1],
                            #             eval_mode = True,
                            #             log_wandb=False,
                            #             save_dir=model_dir,
                            #             title_ext = f"E{epoch}")
                            #   else:
                            #     predict_and_plot_video_displacements(dataset, model, device,
                            #           [start_dates_plot[-1]], n_pred_plot,
                            #           t_step_to_plot,
                            #           lat_points = lat_lon_points[0],
                            #           lon_points= lat_lon_points[1],
                            #           eval_mode = True,
                            #           log_wandb=False,
                            #           save_dir=model_dir,
                            #           title_ext = f"E{epoch}")