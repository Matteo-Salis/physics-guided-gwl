import torch
from tqdm import tqdm
import wandb

from torchview import draw_graph
from utils.plot_ST_MultiPoint import *
from torch import autograd
from loss.losses_ST_MultiPoint import *


def pure_dl_trainer(epoch, dataset, model, train_loader, loss_fn, optimizer, model_dir, model_name,
                      start_dates_plot, n_pred_plot, sensors_to_plot, t_step_to_plot, lat_lon_points,
                      device = "cuda", plot_arch = True, l2_alpha = 0, plot_displacements = False):  
    
    with tqdm(train_loader, unit="batch") as tepoch:
        #with autograd.detect_anomaly():
                    
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
                        
                        optimizer.zero_grad()
                        
                        if "MoE" in model_name:
                            Y_hat, aux_loss = model(X, W, Z, mc_dropout = True,
                                                    get_aux_loss = True)
                            moe = True
                        else:
                            Y_hat = model(X, W, Z, mc_dropout = True)
                            moe = False
                            
                        loss = loss_fn(Y_hat,
                                    Y[0],
                                    Y[1])
                        
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
                        if (epoch+1) % 25 == 0:
                            wandb_time_series(dataset, model, device,
                                start_dates_plot, n_pred_plot,
                                sensors_to_plot,
                                eval_mode = False)
                            
                            if plot_displacements is False:
                                wandb_video(dataset, model, device,
                                            start_dates_plot, n_pred_plot,
                                            t_step_to_plot,
                                            lat_points = lat_lon_points[0],
                                            lon_points= lat_lon_points[1],
                                            eval_mode = False)
                            else:
                                wandb_video_displacements(dataset, model, device,
                                            start_dates_plot, n_pred_plot,
                                            t_step_to_plot,
                                            lat_points = lat_lon_points[0],
                                            lon_points= lat_lon_points[1],
                                            eval_mode = False)
                        
                        if epoch == 0 and plot_arch is True:
                            print("Saving plot of the model's architecture...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (X, W, Z),
                                                                                    device = device)})
                            
                        if (epoch+1) % 50 == 0:
                            
                            print("Computing iterated predictions...")
                            
                            wandb_time_series(dataset, model, device,
                              [start_dates_plot[-1]], n_pred_plot,
                              sensors_to_plot,
                              eval_mode = True)
                            
                            if plot_displacements is False:
                                wandb_video(dataset, model, device,
                                        [start_dates_plot[-1]], n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = True)
                            
                            else:
                                wandb_video_displacements(dataset, model, device,
                                        [start_dates_plot[-1]], n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = True)
                            
                            
def physics_guided_trainer(epoch, dataset, model, train_loader, loss_fn, optimizer, model_dir, model_name,
                      start_dates_plot, n_pred_plot, sensors_to_plot, t_step_to_plot, lat_lon_points,
                      device = "cuda", plot_arch = True,
                      l2_alpha = 0, HC_alpha = 1, coherence_alpha = 1,
                      plot_displacements = False):  
    
    with tqdm(train_loader, unit="batch") as tepoch:
        #with autograd.detect_anomaly():
                    
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
                        
                        optimizer.zero_grad()
                        
                        Y_hat, Displacement_GW, Displacement_S, hydrConductivity = model(X, W, Z, mc_dropout = True,
                                                                                         get_displacement_terms = True)
                            
                        loss = loss_fn(Y_hat,
                                    Y[0],
                                    Y[1])
                        
                        print("Training_data_loss: ", loss.item(), end = " --- ")
                        
                        if l2_alpha > 0:
                            loss += l2_alpha * loss_l2_regularization(model)
                        
                            
                        if HC_alpha > 0:
                            HC_loss = HC_alpha * HydroConductivity_reg(hydrConductivity,
                                                                     denorm_sigma=dataset.norm_factors["target_stds"])
                            loss += HC_loss
                            
                            print("HC_loss: ", HC_loss.item(), end = " --- ")
                            
                        if coherence_alpha > 0:
                            coherence_loss = coherence_alpha * coherence_loss(hydrConductivity,
                                                                    Displacement_GW,
                                                                    Displacement_S)
                            loss += coherence_loss
                            
                            print("COH_loss: ", coherence_loss.item(), end = " --- ")
                            
                        
                        
                        
                        loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Total_loss":loss.item()})   
                        
                    # Plots
                    #model.eval()
                    with torch.no_grad():
                        if (epoch+1) % 25 == 0:
                            wandb_time_series(dataset, model, device,
                                start_dates_plot, n_pred_plot,
                                sensors_to_plot,
                                eval_mode = False)
                            
                            if plot_displacements is False:
                                wandb_video(dataset, model, device,
                                            start_dates_plot, n_pred_plot,
                                            t_step_to_plot,
                                            lat_points = lat_lon_points[0],
                                            lon_points= lat_lon_points[1],
                                            eval_mode = False)
                            else:
                                wandb_video_displacements(dataset, model, device,
                                            start_dates_plot, n_pred_plot,
                                            t_step_to_plot,
                                            lat_points = lat_lon_points[0],
                                            lon_points= lat_lon_points[1],
                                            eval_mode = False)
                        
                        if epoch == 0 and plot_arch is True:
                            print("Saving plot of the model's architecture...")
                            wandb.log({"model_arch": plot_model_graph(model_dir, model_name, model,
                                                                      sample_input = (X, W, Z),
                                                                                    device = device)})
                            
                        if (epoch+1) % 50 == 0:
                            
                            print("Computing iterated predictions...")
                            
                            wandb_time_series(dataset, model, device,
                              [start_dates_plot[-1]], n_pred_plot,
                              sensors_to_plot,
                              eval_mode = True)
                            
                            if plot_displacements is False:
                                wandb_video(dataset, model, device,
                                        [start_dates_plot[-1]], n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = True)
                            
                            else:
                                wandb_video_displacements(dataset, model, device,
                                        [start_dates_plot[-1]], n_pred_plot,
                                        t_step_to_plot,
                                        lat_points = lat_lon_points[0],
                                        lon_points= lat_lon_points[1],
                                        eval_mode = True)