import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import torch
import wandb
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torchview import draw_graph

from collections import deque

from matplotlib.colors import TwoSlopeNorm
    
    
#######################
#### ST MultiPoint ####
#######################
    
def compute_predictions_MultiPoint(date, dataset, model, device, X = None, Z_grid = None, get_Z = False,
                                   get_displacement_terms = False):
    
    subset_df = dataset.lagged_df.loc[pd.IndexSlice[date,:],:]
    subset_df_filled = dataset.lagged_df_filled.loc[pd.IndexSlice[date,:],:]
    
    W = dataset.get_weather_features(date)
        
    if X is None:
        X = dataset.get_lagged_features(subset_df, subset_df_filled)
        
    if Z_grid is None:
        Z = dataset.get_target_st_info(subset_df)
    else:
        Z = torch.from_numpy(Z_grid).to(torch.float32)
        (doy_sin, doy_cos), _ = dataset.temporal_encoding(mode = "sin", dates = pd.DatetimeIndex([date]))
        Z = torch.cat([Z,
                       torch.ones((Z.shape[0],1))*doy_sin,
                       torch.ones((Z.shape[0],1))*doy_cos],
                      dim = -1).to(torch.float32)
        
    Z = Z.to(device)
    Y, _ = dataset.get_target_values(subset_df)
    
    if get_displacement_terms is False:
        Y_hat = model(X = [X[0].unsqueeze(0).to(device),
                        X[1].unsqueeze(0).to(device),
                        X[2].unsqueeze(0).to(device)],
                    W = [W[0].unsqueeze(0).to(device),
                        W[1].unsqueeze(0).to(device)],
                    Z = Z.unsqueeze(0).to(device)
                    )
        
        output = [Y,
            Y_hat]
    
    else:
        
        Y_hat, Displacement_GW, Displacement_S, hydrConductivity, Lag_GW = model(X = [X[0].unsqueeze(0).to(device),
                        X[1].unsqueeze(0).to(device),
                        X[2].unsqueeze(0).to(device)],
                    W = [W[0].unsqueeze(0).to(device),
                        W[1].unsqueeze(0).to(device)],
                    Z = Z.unsqueeze(0).to(device),
                    get_displacement_terms = get_displacement_terms,
                    get_lag_term = get_displacement_terms)
        
        output = [Y,
            Y_hat,
            Displacement_GW,
            Displacement_S,
            hydrConductivity,
            Lag_GW]
    
    if get_Z is True:
        output.append(Z)
    
    return output


def compute_predictions_ST_MultiPoint(dataset, model, device, start_date, n_pred,
                                      Z_grid = None, iter_pred = False,
                                      get_displacement_terms = False):
    
    start_date_idx = dataset.dates.get_loc(start_date)
    
    predictions = []
    true = []
    if get_displacement_terms is True:
        Displacement_GW = []
        Displacement_S = []
        hydrConductivity = []
        Lag_GW = []
    
    if iter_pred is False:
    
        for i in tqdm(range(n_pred)):
            pred_list = compute_predictions_MultiPoint(dataset.dates[start_date_idx+i],
                                            dataset,
                                            model,
                                            device,
                                            Z_grid = Z_grid,
                                            get_Z = False,
                                            get_displacement_terms = get_displacement_terms)
            
            Y = pred_list[0]
            Y_hat = pred_list[1]
            
            predictions.append(Y_hat)
            true.append(Y)
            
            if get_displacement_terms is True:
                Displacement_GW.append(pred_list[2])
                Displacement_S.append(pred_list[3])
                hydrConductivity.append(pred_list[4])
                Lag_GW.append(pred_list[5])
            
    else:
        X = None
        X_deque = [deque(maxlen=len(dataset.target_lags)),
             deque(maxlen=len(dataset.target_lags)),
             deque(maxlen=len(dataset.target_lags))]
        
        for i in tqdm(range(n_pred)):
            pred_list = compute_predictions_MultiPoint(dataset.dates[start_date_idx+i],
                                            dataset,
                                            model,
                                            device,
                                            X = X,
                                            Z_grid = Z_grid,
                                            get_Z = True,
                                            get_displacement_terms = get_displacement_terms)
            Y = pred_list[0]
            Y_hat = pred_list[1]
            Z_pred = pred_list[-1]
            
            X_deque[0].appendleft(Y_hat)
            X_deque[1].appendleft(Z_pred)
            X_deque[2].appendleft(torch.zeros_like(Y_hat).to(torch.bool).to(device))
            
            predictions.append(Y_hat)
            true.append(Y)
            
            if get_displacement_terms is True:
                Displacement_GW.append(pred_list[2])
                Displacement_S.append(pred_list[3])
                hydrConductivity.append(pred_list[4])
                Lag_GW.append(pred_list[5])
            
            if i >= len(dataset.target_lags):
                X = [torch.stack(list(X_deque[0])).to(device),
                     torch.stack(list(X_deque[1])).to(device),
                     torch.stack(list(X_deque[2])).to(device)]
                
        del X_deque
            
    true = torch.stack(true, dim = 0).to(device)
    predictions = torch.stack(predictions, dim = 0).to(device)
    
    output_list = [true, predictions]
    
    if get_displacement_terms is True:
        Displacement_GW = torch.stack(Displacement_GW).to(device)
        output_list.append(Displacement_GW)
        Displacement_S = torch.stack(Displacement_S).to(device)
        output_list.append(Displacement_S)
        hydrConductivity = torch.stack(hydrConductivity).to(device)
        output_list.append(hydrConductivity)
        Lag_GW = torch.stack(Lag_GW).to(device)
        output_list.append(Lag_GW)
    
    return output_list
    
    
def build_ds_from_pred(y, dataset, start_date, n_pred, sensor_names):
    
    start_date_idx = dataset.dates.get_loc(start_date)
    dates = [dataset.dates[start_date_idx+i] for i in range(n_pred)]
    
    pd_ds = pd.DataFrame(data = y,
                         index = dates,
                         columns = sensor_names)
    
    return pd_ds


def plot_time_series(y_hat, y, title,
                   save_dir = None,
                   print_plot = False):
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_hat, label = "Prediction", marker = "o", lw = 0.7, markersize = 2, color = "tab:orange")
    ax.plot(y, label = "Truth", marker = "o", lw = 0.7, markersize = 2, color = "tab:blue")
    
    ax.tick_params(axis='x', rotation=40)
    ax.set_ylabel("H [m]")

    ax.set_title(title)

    ax.legend()
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    if print_plot is True:
        plt.show()
        
    else:
        return fig
    
    
def grid_generation(dataset, nh = 30, hw = 45, bbox = None):
    if bbox is None:
        
        bbox = [dataset.dtm_roi.x.min().values,
        dataset.dtm_roi.x.max().values,
        dataset.dtm_roi.y.min().values,
        dataset.dtm_roi.y.max().values]

    x = np.linspace(bbox[0], bbox[1], hw)
    y = np.linspace(bbox[2], bbox[3], nh)[::-1]
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    coords = np.stack([Y, X], axis = -1)


    dtm_grid = dataset.dtm_roi.sel(x = x, y = y,
                    method = "nearest").values #(1,H,W)

    if dataset.config["normalization"] is True:
        coords[:,:,0] = (coords[:,:,0] - dataset.norm_factors["lat_mean"])/dataset.norm_factors["lat_std"]
        coords[:,:,1] = (coords[:,:,1] - dataset.norm_factors["lon_mean"])/dataset.norm_factors["lon_std"]

    Z = np.concat([coords, np.moveaxis(dtm_grid, 0, -1)], axis=-1)

    Z = Z.reshape(Z.shape[0]*Z.shape[1], Z.shape[2])
    
    return Z
    
#### WANDB ######

def predict_and_plot_time_series(dataset, model, device,
                    start_dates_input, n_pred,
                    sensors_to_plot,
                    eval_mode,
                    log_wandb = False,
                    save_dir = None,
                    title_ext = ""):
    
        title_ext += "_iter" if eval_mode else ""
        
        for date in start_dates_input:
    
            
            true, predictions = compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                  np.datetime64(date),
                                                                  n_pred,
                                                                  iter_pred = eval_mode)
            
            prediction_ds = build_ds_from_pred(predictions.detach().cpu(), dataset, start_date=np.datetime64(date), n_pred=n_pred, sensor_names=dataset.sensor_id_list)
            true_ds = build_ds_from_pred(true.detach().cpu(), dataset, start_date=np.datetime64(date), n_pred=n_pred, sensor_names=dataset.sensor_id_list)
           
            # Denormalization
            if dataset.config["normalization"] is True and dataset.config["target_norm_type"] is not None:
                prediction_ds = (prediction_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
                true_ds = (true_ds * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
            
                                      
            for sensor_id in sensors_to_plot:
                
                municipality = dataset.wtd_geodf["munic"].loc[dataset.wtd_geodf["sensor_id"] == sensor_id].values[0]
                
                if log_wandb is True:
                    wandb.log({f"{municipality}_ts_{date}_{title_ext}":wandb.Image(plot_time_series(
                                                                                prediction_ds[sensor_id], true_ds[sensor_id],
                                                                                title = f"Prediction {title_ext} {sensor_id} - {municipality} - from {date}",
                                                                                save_dir = None,
                                                                                print_plot = False))})
                
                else:
                    plot_time_series(prediction_ds[sensor_id], true_ds[sensor_id],
                                    title = f"Prediction {title_ext} {sensor_id} - {municipality} - from {date}",
                                    save_dir = f"{save_dir}/ts_plots/{municipality}_{date}_{title_ext}",
                                    print_plot = False)
                    
                plt.close("all")
                
            del prediction_ds
            del true_ds
                
                
def predict_and_plot_video(dataset, model, device,
                start_dates_input, n_pred,
                t_step_to_plot,
                lat_points,
                lon_points,
                eval_mode,
                log_wandb = False,
                save_dir = None,
                title_ext = ""):
    

        Z_grid = grid_generation(dataset, lat_points,lon_points)
        Z_grid_matrix = Z_grid.reshape(lat_points,lon_points,3)
        title_ext += "_iter" if eval_mode else ""
        
        for date in start_dates_input:
    
            
            _, predictions = compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                  np.datetime64(date),
                                                                  n_pred,
                                                                  Z_grid = Z_grid,
                                                                  iter_pred = eval_mode)
            
            predictions_grid = predictions.reshape(n_pred,lat_points,lon_points).detach().cpu()
            
            # Denormalization
            if dataset.config["normalization"] is True:
                Z_grid_matrix_lat = (Z_grid_matrix[:,:,0]*dataset.norm_factors["lat_std"]) + dataset.norm_factors["lat_mean"]
                Z_grid_matrix_lon = (Z_grid_matrix[:,:,1]*dataset.norm_factors["lon_std"]) + dataset.norm_factors["lon_mean"]
                dtm = (Z_grid_matrix[:,:,2]*dataset.norm_factors["dtm_std"].values) + dataset.norm_factors["dtm_mean"].values
                
                if dataset.config["target_norm_type"] is not None:
                    predictions_grid = (predictions_grid * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
            else:
                Z_grid_matrix_lat = Z_grid_matrix[:,:,0]
                Z_grid_matrix_lon = Z_grid_matrix[:,:,1]
                dtm = Z_grid_matrix[:,:,2]
        
            
            
            # Build xr
            start_date_idx = dataset.dates.get_loc(np.datetime64(date))
            date_seq = [dataset.dates[start_date_idx+i] for i in range(n_pred)]
            
            predictions_xr = xarray.DataArray(data = predictions_grid,
                                coords = dict(
                                            lat=("lat", Z_grid_matrix_lat[:,0]),
                                            lon=("lon", Z_grid_matrix_lon[0,:]),
                                            time=date_seq),
                                dims = ["time","lat", "lon"]
                                )
            
            if dataset.config["piezo_head"] is True and dataset.config["relative_target"] is False:
                predictions_xr_wtd = dtm - predictions_xr
                
            else:
                print("Can't produce wtd... aborting map plots!")
                return
                
            vmin_H = predictions_xr.min().values
            vmax_H = predictions_xr.max().values
            vmin_WTD = predictions_xr_wtd.min().values
            vmax_WTD = predictions_xr_wtd.max().values
                                      
            for t_step in t_step_to_plot:
                
                if log_wandb is True:
                    wandb.log({f"map_prediction_{title_ext}_{date} -":wandb.Image(
                        plot_map(predictions_xr[t_step], predictions_xr_wtd[t_step],
                                title = f"Map prediction {title_ext}",
                                shapefile=dataset.piemonte_shp,
                                vmin = [vmin_H,vmin_WTD],
                                vmax = [vmax_H,vmax_WTD],
                                save_dir = None,
                                print_plot = False))})
                else:
                    plot_map(predictions_xr[t_step], predictions_xr_wtd[t_step],
                                title = f"Map prediction {title_ext}",
                                shapefile=dataset.piemonte_shp,
                                vmin = [vmin_H,vmin_WTD],
                                vmax = [vmax_H,vmax_WTD],
                                save_dir = f"{save_dir}/map_plots/map_{title_ext}_{date}",
                                print_plot = False)
                    
                plt.close("all")
                
            # delete object
            del predictions_xr
            del predictions_xr_wtd
                    
                
                
def predict_and_plot_video_displacements(dataset, model, device,
                start_dates_input, n_pred,
                t_step_to_plot,
                lat_points,
                lon_points,
                eval_mode,
                log_wandb = False,
                save_dir = None,
                title_ext = ""):
    

        Z_grid = grid_generation(dataset, lat_points,lon_points)
        Z_grid_matrix = Z_grid.reshape(lat_points,lon_points,3)
        title_ext += "_iter" if eval_mode else ""
        
        for date in start_dates_input:
    
            
            _, predictions, displacement_gw, displacement_s, hydrConductivity, _ = compute_predictions_ST_MultiPoint(dataset, model, device,
                                                                  np.datetime64(date),
                                                                  n_pred,
                                                                  Z_grid = Z_grid,
                                                                  iter_pred = eval_mode,
                                                                  get_displacement_terms = True)
            
            predictions_grid = predictions.reshape(n_pred,lat_points,lon_points).detach().cpu()
            displacement_gw_grid = displacement_gw.reshape(n_pred,lat_points,lon_points).detach().cpu()
            displacement_s_grid = displacement_s.reshape(n_pred,lat_points,lon_points).detach().cpu()
            hydrConductivity_grid = hydrConductivity.reshape(n_pred,lat_points,lon_points).detach().cpu()
            
            # Denormalization
            if dataset.config["normalization"] is True:
                Z_grid_matrix_lat = (Z_grid_matrix[:,:,0]*dataset.norm_factors["lat_std"]) + dataset.norm_factors["lat_mean"]
                Z_grid_matrix_lon = (Z_grid_matrix[:,:,1]*dataset.norm_factors["lon_std"]) + dataset.norm_factors["lon_mean"]
                dtm = (Z_grid_matrix[:,:,2]*dataset.norm_factors["dtm_std"].values) + dataset.norm_factors["dtm_mean"].values
                
                if dataset.config["target_norm_type"] is not None:
                    predictions_grid = (predictions_grid * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
                    displacement_gw_grid = displacement_gw_grid * dataset.norm_factors["target_stds"]
                    displacement_s_grid = displacement_s_grid * dataset.norm_factors["target_stds"]
                    hydrConductivity_grid = hydrConductivity_grid * dataset.norm_factors["target_stds"]
            else:
                Z_grid_matrix_lat = Z_grid_matrix[:,:,0]
                Z_grid_matrix_lon = Z_grid_matrix[:,:,1]
                dtm = Z_grid_matrix[:,:,2]
        
            
            
            # Build xr
            start_date_idx = dataset.dates.get_loc(np.datetime64(date))
            date_seq = [dataset.dates[start_date_idx+i] for i in range(n_pred)]
            
            predictions_xr = xarray.DataArray(data = predictions_grid,
                                coords = dict(
                                            lat=("lat", Z_grid_matrix_lat[:,0]),
                                            lon=("lon", Z_grid_matrix_lon[0,:]),
                                            time=date_seq),
                                dims = ["time","lat", "lon"]
                                )
            
            displacement_gw_xr = xarray.DataArray(data = displacement_gw_grid,
                                coords = dict(
                                            lat=("lat", Z_grid_matrix_lat[:,0]),
                                            lon=("lon", Z_grid_matrix_lon[0,:]),
                                            time=date_seq),
                                dims = ["time","lat", "lon"]
                                )
            
            displacement_s_xr = xarray.DataArray(data = displacement_s_grid,
                                coords = dict(
                                            lat=("lat", Z_grid_matrix_lat[:,0]),
                                            lon=("lon", Z_grid_matrix_lon[0,:]),
                                            time=date_seq),
                                dims = ["time","lat", "lon"]
                                )
            
            hydrConductivity_xr = xarray.DataArray(data = hydrConductivity_grid,
                                coords = dict(
                                            lat=("lat", Z_grid_matrix_lat[:,0]),
                                            lon=("lon", Z_grid_matrix_lon[0,:]),
                                            time=date_seq),
                                dims = ["time","lat", "lon"]
                                )
            
            if dataset.config["piezo_head"] is True and dataset.config["relative_target"] is False:
                predictions_xr_wtd = dtm - predictions_xr
                
            else:
                print("Can't produce wtd... aborting map plots!")
                return
                
            vmin_H = predictions_xr.min().values
            vmax_H = predictions_xr.max().values
            vmin_WTD = predictions_xr_wtd.min().values
            vmax_WTD = predictions_xr_wtd.max().values
                                      
            for t_step in t_step_to_plot:
                
                if log_wandb is True:
                
                    wandb.log({f"map_prediction_{title_ext}_{date} -":wandb.Image(
                        plot_pred_displacement_maps(predictions_xr[t_step],
                                                    predictions_xr_wtd[t_step],
                                                    displacement_gw_xr[t_step],
                                                    displacement_s_xr[t_step],
                                                    hydrConductivity_xr[t_step],
                                                    title = f"Map prediction {title_ext}",
                                                    shapefile=dataset.piemonte_shp,
                                                    vmin = [vmin_H,vmin_WTD],
                                                    vmax = [vmax_H,vmax_WTD],
                                                    save_dir = None,
                                                    print_plot = False))})
                else:
                    
                    plot_pred_displacement_maps(predictions_xr[t_step],
                                                    predictions_xr_wtd[t_step],
                                                    displacement_gw_xr[t_step],
                                                    displacement_s_xr[t_step],
                                                    hydrConductivity_xr[t_step],
                                                    title = f"Map prediction {title_ext}",
                                                    shapefile=dataset.piemonte_shp,
                                                    vmin = [vmin_H,vmin_WTD],
                                                    vmax = [vmax_H,vmax_WTD],
                                                    save_dir = f"{save_dir}/map_plots/map_{title_ext}_{date}",
                                                    print_plot = False)
                    
                plt.close("all")
                
            # delete object
            del predictions_xr
            del predictions_xr_wtd
            del displacement_gw_xr
            del displacement_s_xr
            del hydrConductivity_xr
        
        

    
def plot_map(predictions_xr,
             predictions_xr_wtd,
                 title,
                 shapefile,
                 vmin = None,
                 vmax = None,
                 save_dir = None, 
                 print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    fig.suptitle(title)
    
    if vmin is None:
        vmin_H = predictions_xr.min().values
        vmin_WTD = predictions_xr_wtd.min().values
    else:
        vmin_H = vmin[0]
        vmin_WTD = vmin[1]
    
    if vmax is None:
        vmax_H = predictions_xr.max().values
        vmax_WTD = predictions_xr_wtd.max().values
    else:
        vmax_H = vmax[0]
        vmax_WTD = vmax[1]

    predictions_xr.plot(ax = ax[0],
                vmin = vmin_H,
                vmax = vmax_H,
                cbar_kwargs={"shrink": 0.9})
    
    shapefile.boundary.plot(ax = ax[0],
                               color = "black",
                               label = "Piedmont's bounds")
    
    ax[0].set_title("Prediction H [m]")
    
    predictions_xr_wtd.plot(ax = ax[1],
                vmin = vmin_WTD,
                vmax = vmax_WTD,
                cbar_kwargs={"shrink": 0.9})
    
    shapefile.boundary.plot(ax = ax[1],
                               color = "black",
                               label = "Piedmont's bounds")
    
    ax[1].set_title("Prediction WTD [m]")

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.show()
        
    else:
        return fig 

def plot_pred_displacement_maps(predictions_xr,
            predictions_xr_wtd,
            displacement_gw_xr,
            displacement_s_xr,
            hydrConductivity_xr,
            title,
            shapefile,
            vmin = None,
            vmax = None,
            save_dir = None, 
            print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,5, figsize = (12,2.25))
    fig.suptitle(title)
    
    if vmin is None:
        vmin_H = predictions_xr.min().values
        vmin_WTD = predictions_xr_wtd.min().values
    else:
        vmin_H = vmin[0]
        vmin_WTD = vmin[1]
    
    if vmax is None:
        vmax_H = predictions_xr.max().values
        vmax_WTD = predictions_xr_wtd.max().values
    else:
        vmax_H = vmax[0]
        vmax_WTD = vmax[1]

    im0 = predictions_xr.plot(ax = ax[0],
                vmin = vmin_H,
                vmax = vmax_H,
                cbar_kwargs={"shrink": 0.9})
    
    shapefile.boundary.plot(ax = ax[0],
                               color = "black",
                               label = "Piedmont's bounds")
    ax[0].set_title("Prediction H [m]")
    ax[0].tick_params(labelsize=6)  # Set tick label size
    
    
    im1 = predictions_xr_wtd.plot(ax = ax[1],
                vmin = vmin_WTD,
                vmax = vmax_WTD,
                cbar_kwargs={"shrink": 0.9})
    
    shapefile.boundary.plot(ax = ax[1],
                               color = "black",
                               label = "Piedmont's bounds")
    ax[1].set_title("Prediction WTD [m]")
    ax[1].tick_params(labelsize=6)  # Set tick label size
    
    norm = TwoSlopeNorm(vcenter=0)
    
    im2 = displacement_gw_xr.plot(ax = ax[2],
                cmap = "seismic_r",# norm = norm
                cbar_kwargs={"shrink": 0.9})
    shapefile.boundary.plot(ax = ax[2],
                               color = "black",
                               label = "Piedmont's bounds")
    ax[2].set_title("Displacement - H [m]")
    ax[2].tick_params(labelsize=6)  # Set tick label size
    
    im3 = displacement_s_xr.plot(ax = ax[3],
                cmap = "seismic_r",# norm = norm
                cbar_kwargs={"shrink": 0.9})
    shapefile.boundary.plot(ax = ax[3],
                               color = "black",
                               label = "Piedmont's bounds")
    ax[3].set_title("Displacement - S [m]")
    ax[3].tick_params(labelsize=6)  # Set tick label size
    
    im4 = hydrConductivity_xr.plot(ax = ax[4],
                                   cbar_kwargs={"shrink": 0.9})
    shapefile.boundary.plot(ax = ax[4],
                               color = "black",
                               label = "Piedmont's bounds")
    ax[4].set_title("hydrConductivity [m/w]")
    # cbar = plt.colorbar(im4, ax = ax[4], fraction=0.05, pad=0.04)
    ax[4].tick_params(labelsize=6)  # Set tick label size

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.show()
        
    else:
        return fig 


###############################
### MAP Plot for Comparisons ###
###############################


def plot_map_all_models(predictions_xr_list,
            title,
            shapefile,
            model_names,
            var_name_title = "H [m]",
            save_dir = None, 
            print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,len(predictions_xr_list), figsize = (10,4))
    fig.suptitle(title)

    for model_i in range(len(predictions_xr_list)):
        predictions_xr_list[model_i].plot(ax = ax[model_i],
                                          cbar_kwargs={"shrink": 0.75})
    
        shapefile.boundary.plot(ax = ax[model_i],
                                color = "black",
                                label = "Piedmont's bounds")
    
        ax[model_i].set_title(f"{model_names[model_i]} {var_name_title}")

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight',
                    dpi = 400, transparent = True)
    
    if print_plot is True:
        plt.show()
        
    else:
        return fig
    
    
def plot_displacement_all_models(displacement_pred_list,
            title,
            shapefile,
            model_names,
            save_dir = None, 
            print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(len(displacement_pred_list),
                           3, figsize = (10,5))
    fig.suptitle(title)

    for model_i in range(len(displacement_pred_list)):
        
        ### Displacement H
        im0 = displacement_pred_list[model_i][0].plot(ax = ax[model_i,0],
                cmap = "seismic_r",# norm = norm
                cbar_kwargs={"shrink": 0.75})
        shapefile.boundary.plot(ax = ax[model_i,0],
                                color = "black",
                                label = "Piedmont's bounds")
        ax[model_i,0].set_title(r"{} $\Delta_{{GW}}$ [m]".format(model_names[model_i]))
        #cbar = plt.colorbar(im0, ax = ax[model_i,0], fraction=0.05, pad=0.04)
        ax[model_i,0].tick_params(labelsize=6)  # Set tick label size
        
        ### Displacement S
        im1 = displacement_pred_list[model_i][1].plot(ax = ax[model_i,1],
            cmap = "seismic_r",# norm = norm
            cbar_kwargs={"shrink": 0.75})
        shapefile.boundary.plot(ax = ax[model_i,1],
                                color = "black",
                                label = "Piedmont's bounds")
        ax[model_i,1].set_title(r"{} $\Delta_S$ [m]".format(model_names[model_i]))
        #cbar = plt.colorbar(im1, ax = ax[model_i,1], fraction=0.05, pad=0.04)
        ax[model_i,1].tick_params(labelsize=6)  # Set tick label size
        
        ### Conductivity
        im2 = displacement_pred_list[model_i][2].plot(ax = ax[model_i,2],
                                                      cbar_kwargs={"shrink": 0.75})
        shapefile.boundary.plot(ax = ax[model_i,2],
                                    color = "black",
                                    label = "Piedmont's bounds")
        ax[model_i,2].set_title(r"{} K [m/w]".format(model_names[model_i]))
        #cbar = plt.colorbar(im2, ax = ax[model_i,2], fraction=0.05, pad=0.04)
        ax[model_i,2].tick_params(labelsize=6)  # Set tick label size


    plt.tight_layout(pad=1.0, h_pad=0.25, w_pad=0.5)
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight',
                    dpi = 400, transparent = True)
    
    if print_plot is True:
        plt.plot()
        
    else:
        return fig 
    
#######################
##### GIF #############
#######################

def generate_gif_from_xr(start_date, n_pred,
                       xr,
                       title,
                       freq,
                       save_dir = None,
                       print_plot = False):
    
    def update_h_wtd_maps(i):
        
        sample_date_i = np.datetime64(start_date) + np.timedelta64(i, freq)
        
        fig.suptitle(f"t0: {start_date} - lead time: {i} {sample_date_i} ",
                     x=0.45, ha="center", y=0.1)
        
        ax.set_title(title)
        
        image.set_array(xr[i,:,:])
        
        return image

    fig, ax = plt.subplots(1,1, figsize = (7,5) )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2, top = 0.9, left=0.1)          # leave space for suptitle
    
    fig.suptitle(f"t0: {start_date} - lead time: {0}",
                 x=0.45, ha="center", y=0.1)
    image = xr[0,:,:].plot(ax = ax, animated=True,
                                                vmin = xr.min().values,
                                                vmax = xr.max().values)
    ax.set_title(title)


    
        
        ## Plot the maps
        
        
    ani = animation.FuncAnimation(fig, update_h_wtd_maps, repeat=True, frames=n_pred, interval=1)

    writer = animation.PillowWriter(fps=1,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800,
                                    )
    
    if save_dir:
        ani.save(f'{save_dir}.gif', writer=writer,
                 dpi=400)
        
    if print_plot is True:
        plt.show()
    
    
###### OLD ##########

def plot_K_lat_lon_maps(K_lat_lon, save_dir = None, print_plot = False):
    ## Plot the maps
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    fig.suptitle(f"Hydraulic Conductivity")
    
    vmin = K_lat_lon.min().values
    vmax = K_lat_lon.min().values

    K_lat_lon.sel(bands = "K_lat").plot(ax = ax[0], vmin = vmin, vmax = vmax)
    ax[0].set_title("K_lat")
    
    K_lat_lon.sel(bands = "K_lon").plot(ax = ax[1], vmin = vmin, vmax = vmax)
    ax[1].set_title("K_lon")

    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
        
    else:
        return fig 
    

def plot_h_wtd_maps(sample_h, sample_wtd, 
                 sample_date, pred_timestep,
                 save_dir = None, 
                 print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    fig.suptitle(f"t0: {sample_date} - Prediction Timestep {pred_timestep}")

    sample_h[pred_timestep,:,:].plot(ax = ax[0])
    ax[0].set_title("Piezometric head")


    sample_wtd[pred_timestep,:,:].plot(ax = ax[1], vmin = sample_wtd.min().values,
                            vmax = sample_wtd.max().values)
    ax[1].set_title("WTD")

    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
        
    else:
        return fig 
    
def plot_sensor_ts(sensor_ds, title,
                   save_dir = None,
                   print_plot = False):
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sensor_ds, label = sensor_ds.columns, marker = "o", lw = 0.7, markersize = 2)
    ax.tick_params(axis='x', rotation=40)

    ax.set_title(title)

    ax.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    if print_plot is True:
        plt.show()
        
    else:
        return fig
    
    
def find_munic_lat_lon_sensor(dataset, sensor_id):
    
    municipality = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor_id, "munic"].values[0]
    lat = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor_id].geometry.y.values[0]
    lon = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor_id].geometry.x.values[0]
    
    return municipality, lat, lon

def find_sensor_pred_in_xr(true_xr, pred_xr, lat, lon):
    ts_pred = pd.DataFrame({"Truth": true_xr.sel(lon  = lon, lat = lat, method = "nearest"),
                       "Prediction": pred_xr.sel(lon  = lon, lat = lat, method = "nearest").values,
                       },
                       index = pred_xr.sel(lon  = lon, lat = lat, method = "nearest").time.values)
    
    return ts_pred

                
###### GIF #######

# def generate_gif_h_wtd(start_date, twindow,
#                        sample_h, sample_wtd,
#                        freq,
#                        save_dir = None,
#                        print_plot = False):

#     fig, ax = plt.subplots(1,2, figsize = (10,4))

#     ax[0].set_title("Piezometric head [m]")
#     ax[1].set_title("WTD [m]")


#     fig.suptitle(f"t0: {start_date} - Prediction Timestep {0}")
#     piezo_image = sample_h[0,:,:].plot(ax = ax[0], animated=True,
#                                                 vmin = sample_h.min().values,
#                                                 vmax = sample_h.max().values,
#                                                 cmap = "Blues")
#     wtd_image = sample_wtd[0,:,:].plot(ax = ax[1], animated=True, 
#                                                 vmin = sample_wtd.min().values,
#                                                 vmax = sample_wtd.max().values,
#                                                 cmap = "Greys")


#     def update_h_wtd_maps(i):
        
#         sample_date_i = np.datetime64(start_date) + np.timedelta64(i+1, freq)
#         fig.suptitle(f"t0: {start_date} - Prediction Timestep {i}: {sample_date_i} ")
        
#         ax[0].set_title("Piezometric head")
#         ax[1].set_title("WTD")
        
#         piezo_image.set_array(sample_h[i,:,:])
#         wtd_image.set_array(sample_wtd[i,:,:])
        
#         return (piezo_image, wtd_image)
        
#         ## Plot the maps
        
        
#     ani = animation.FuncAnimation(fig, update_h_wtd_maps, repeat=True, frames=twindow, interval=1)

#     writer = animation.PillowWriter(fps=5,
#                                     metadata=dict(artist='Me'),
#                                     bitrate=1800)
    
#     if save_dir:
#         ani.save(f'{save_dir}.gif', writer=writer)
        
#     if print_plot is True:
#         plt.show()
                
        

################
#### COMMON ####
################

                            
def plot_model_graph(file_path, file_name, model, sample_input, device, depth = 1):
    
    model_graph = draw_graph(model, input_data=sample_input, device=device, depth = depth, mode = "train")
    model_graph.visual_graph.render(format='png', filename = file_name, directory= f"{file_path}/")
    model_arch = wandb.Image(f"{file_path}/{file_name}.png", caption="model's architecture")
    
    return model_arch
            