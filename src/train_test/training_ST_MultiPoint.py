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
                        
                        print("Training_data_loss: ", loss.item(), end = " -- ")
                        
                        if l2_alpha > 0:
                            l2reg_loss = loss_l2_regularization(model)
                            loss += l2_alpha * l2reg_loss
                            print("L2_loss: ", l2reg_loss.item(), end = " -- ")
                        
                        if moe is True:
                            loss += aux_loss
                            print("aux_moe_loss: ", aux_loss, end = " -- ")
                            
                        print("Total_loss: ", loss.item())
                        
                        loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Training_Total_loss":loss.item()})   
                        
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
                    tstep_control_points,
                    device = "cuda", plot_arch = True,
                    l2_alpha = 0, coherence_alpha = 1, diffusion_alpha = 1,
                    reg_diffusion_alpha = 0, reg_displacement_S = 0,
                    reg_latlon_smoothness=0, reg_temp_smoothness = 0,
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
                        
                        Y_hat, Displacement_GW, Displacement_S, HydrConductivity, Lag_GW_hat = model(X, W, Z, mc_dropout = True,
                                      get_displacement_terms = True, get_lag_term = True)
                            
                        loss = loss_fn(Y_hat,
                                    Y[0],
                                    Y[1])
                        
                        print("Training_data_loss: ", round(loss.item(),7), end = " -- ")
                        
                        if l2_alpha > 0:
                            loss += l2_alpha * loss_l2_regularization(model)
                        
                        if coherence_alpha > 0:
                            coh_loss = coherence_alpha * coherence_loss(X[0],
                                                                        X[2],
                                                                        "mse",
                                                                        Lag_GW_hat)
                            
                            loss += coh_loss
                            
                            print(f"COH_loss: {round(coh_loss.item(),7)}", end = " -- ")
                            wandb.log({"COH_loss":coh_loss.item()})
                            
                        if reg_diffusion_alpha>0:
                                reg_diffusion_loss = reg_diffusion_alpha * displacement_reg(Displacement_GW/(HydrConductivity+1e-5),
                                                                        res_fn = "mse")
                            
                                loss += reg_diffusion_loss
                                
                                print(f"RegD_loss: {round(reg_diffusion_loss.item(),7)}", end = " -- ")
                                wandb.log({"RegD_loss":reg_diffusion_loss.item()})
                                
                        
                        if reg_displacement_S>0:
                                reg_displacement_S_loss = reg_displacement_S * displacement_reg(Displacement_S,
                                                                        res_fn = "mse")
                            
                                loss += reg_displacement_S_loss
                                
                                print(f"RegS_loss: {round(reg_displacement_S_loss.item(),7)}", end = " -- ")
                                wandb.log({"RegS_loss":reg_displacement_S_loss.item()})
                            
                            
                            
                        ### Control Points Losses
                             
                        if reg_temp_smoothness+reg_latlon_smoothness+diffusion_alpha>0:
                            ## Prediction
                            Y_hat_CP, Displacement_GW_CP, Displacement_S_CP, HydrConductivity_CP, Lag_GW_CP = Control_Points_Predictions(dataset, model, device,
                                tstep_control_points, lat_points = lat_lon_points[0], lon_points = lat_lon_points[1],
                                eval_mode = False)
                            
                            if diffusion_alpha > 0:
                                diff_loss = diffusion_alpha * diffusion_loss(
                                                        Lag_GW = Lag_GW_CP[:,-1,:].reshape(tstep_control_points,
                                                                                lat_lon_points[0],
                                                                                lat_lon_points[1]),
                                                        Displacement_GW = Displacement_GW_CP.reshape(tstep_control_points,
                                                                                lat_lon_points[0],
                                                                                lat_lon_points[1]),
                                                        K = HydrConductivity_CP.reshape(tstep_control_points,
                                                                                lat_lon_points[0],
                                                                                lat_lon_points[1]),
                                                        res_fn = "mse",
                                                        normf_mu = dataset.norm_factors["target_means"],
                                                        normf_sigma = dataset.norm_factors["target_stds"],
                                                        )
                                
                                loss += diff_loss
                                
                                print(f"CP_Diff_loss: {round(diff_loss.item(),7)}", end = " -- ")
                                wandb.log({"CP_Diff_loss":diff_loss.item()})
                            
                            if reg_diffusion_alpha>0:
                                reg_diffusion_loss = reg_diffusion_alpha * displacement_reg(Displacement_GW_CP/(HydrConductivity_CP+1e-5),
                                                                        res_fn = "mse")
                            
                                loss += reg_diffusion_loss
                                
                                print(f"CP_RegD_loss: {round(reg_diffusion_loss.item(),7)}", end = " -- ")
                                wandb.log({"CP_RegD_loss":reg_diffusion_loss.item()})
                                
                            if reg_displacement_S>0:
                                reg_displacement_S_loss = reg_displacement_S * displacement_reg(Displacement_S_CP,
                                                                        res_fn = "mse")
                            
                                loss += reg_displacement_S_loss
                                
                                print(f"CP_RegS_loss: {round(reg_displacement_S_loss.item(),7)}", end = " -- ")
                                wandb.log({"CP_RegS_loss":reg_displacement_S_loss.item()})
                                
                            if reg_latlon_smoothness >0:
                                reg_smoothness_latlon_dis_gw = reg_latlon_smoothness * sum(smoothness_reg(Displacement_GW_CP.reshape(tstep_control_points,
                                                                                lat_lon_points[0],
                                                                                lat_lon_points[1]), mode = "lon_lat"))
                                reg_smoothness_latlon_dis_s = reg_latlon_smoothness * sum(smoothness_reg(Displacement_S_CP.reshape(tstep_control_points,
                                                                                lat_lon_points[0],
                                                                                lat_lon_points[1]), mode = "lon_lat"))
                                
                                # reg_smoothness_latlon_pred = reg_latlon_smoothness * sum(smoothness_reg(Y_hat_CP.reshape(tstep_control_points,
                                #                                                 lat_lon_points[0],
                                #                                                 lat_lon_points[1]), mode = "lon_lat"))
                                
                                print(f"LatLon Smooth: {round(reg_smoothness_latlon_dis_gw.item(),5)}; {round(reg_smoothness_latlon_dis_s.item(),5)}", end = " -- ")
                                loss += reg_smoothness_latlon_dis_gw + reg_smoothness_latlon_dis_s
                                
                            if reg_temp_smoothness > 0:
                                
                                reg_smoothness_temp_pred = reg_temp_smoothness * smoothness_reg(Y_hat_CP, mode = "temp")
                                print(f"Temp Smooth: {round(reg_smoothness_temp_pred.item(),5)}", end = " -- ")
                                loss += reg_smoothness_temp_pred
                                
                                
                        print("Total_loss: ", round(loss.item(),7))
                        
                        loss.backward()
                        optimizer.step()
                        
                        wandb.log({"Training_Total_loss":loss.item()})   
                        
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
                                
                                
                                

def Control_Points_Predictions(dataset, model, device,
                               tstep_control_points,
                               lat_points, lon_points,
                               eval_mode = False):
    
    ###Z_grid mettiamo in load_trainer con partial, anche lat_ponints e lon:points
    ### non necessario neanche reshape a questo stadio, se facciamo vincolo su diffusion poi sì
    
    
    # Sample a random dates
    random_idx = torch.randint(low=0, high=len(dataset.dates)-tstep_control_points, size=(1,)).item()
    random_date = dataset.dates[random_idx]
    
    Z_grid = grid_generation(dataset, lat_points,lon_points)
    
    _, predictions, displacement_gw, displacement_s, hydrConductivity, Lag_GW = compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                  np.datetime64(random_date),
                                                                  tstep_control_points,
                                                                  Z_grid = Z_grid,
                                                                  iter_pred = eval_mode,
                                                                  get_displacement_terms = True)
            
    # predictions_grid = predictions.reshape(tstep_control_points,lat_points,lon_points)
    # displacement_gw_grid = displacement_gw.reshape(tstep_control_points,lat_points,lon_points)
    # displacement_s_grid = displacement_s.reshape(tstep_control_points,lat_points,lon_points)
    # hydrConductivity_grid = hydrConductivity.reshape(tstep_control_points,lat_points,lon_points)
    #Z_grid_matrix = Z_grid.reshape(lat_points,lon_points,3)
    
    return predictions, displacement_gw, displacement_s, hydrConductivity, Lag_GW
    
    # # Denormalization
    # if dataset.config["normalization"] is True:
    #     Z_grid_matrix_lat = (Z_grid_matrix[:,:,0]*dataset.norm_factors["lat_std"]) + dataset.norm_factors["lat_mean"]
    #     Z_grid_matrix_lon = (Z_grid_matrix[:,:,1]*dataset.norm_factors["lon_std"]) + dataset.norm_factors["lon_mean"]
    #     dtm = (Z_grid_matrix[:,:,2]*dataset.norm_factors["dtm_std"].values) + dataset.norm_factors["dtm_mean"].values
        
    #     if dataset.config["target_norm_type"] is not None:
    #         predictions_grid = (predictions_grid * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
    #         displacement_gw_grid = displacement_gw_grid * dataset.norm_factors["target_stds"]
    #         displacement_s_grid = displacement_s_grid * dataset.norm_factors["target_stds"]
    #         hydrConductivity_grid = hydrConductivity_grid * dataset.norm_factors["target_stds"]
    # else:
    #     Z_grid_matrix_lat = Z_grid_matrix[:,:,0]
    #     Z_grid_matrix_lon = Z_grid_matrix[:,:,1]
    #     dtm = Z_grid_matrix[:,:,2]
