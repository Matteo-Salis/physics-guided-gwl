import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import torch
import wandb
import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from rasterio.enums import Resampling

from torchview import draw_graph

################
##### 1-D ######
################

def predict_series_points(ds, date_t0, sensor_number, model, device):

    sample_idx = ds.get_iloc_from_date(date_max = date_t0) + 1

    x, z, w_values, y, x_mask, y_mask = ds[sample_idx + sensor_number]

    x = x.to(device)
    x_mask = x_mask.to(device)
    z = z.to(device)
    w = [w_values.to(device), ds.weather_coords_dtm.to(device)]
    y = y.to(device)
    y_mask = y_mask.to(device)

    y_hat = model(x, z, w, x_mask)

    date_t0 = ds.wtd_df.iloc[sample_idx + sensor_number].name[0]
    sensor = ds.wtd_df.iloc[sample_idx + sensor_number].name[1]


    dates = pd.date_range(date_t0+ np.timedelta64(1, "D"),
                                    date_t0 + np.timedelta64(180, "D"),
                                    freq="D")

    df = pd.DataFrame({"prediction": y_hat.detach().cpu().numpy(),
                            "true": y.detach().cpu().numpy()}, index = dates )

    df["prediction"] = (df["prediction"] * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]
    df["true"] = (df["true"] * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]

    return [df, sensor]

def plot_one_series(ds, date_t0, df_prediction = None, sensor = None,
                    save_dir = None, print_plot = False,
                    generate_points = True, model = None, device = None):
    
    """
    if generate_point = True provide the sensor number, otherwise the sensor_id
    """
    
    if generate_points is True:
        df_prediction, sensor_id = predict_series_points(ds, date_t0, sensor, model, device)

    fig, ax = plt.subplots()
    ax.plot(df_prediction, label = df_prediction.columns, marker = "o", lw = 0.7, markersize = 2)

    municipality = ds.wtd_names["munic"].loc[ds.wtd_names["sensor_id"] == sensor_id].values[0]
    ax.set_title(f"{sensor_id} - {municipality} - from {date_t0}")

    ax.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    if print_plot is True:
        plt.show()
        
    else:
        return fig

############## MAP ##############

def predict_map_points(ds, lon_point, 
                 sample_date,
                 model, device,
                 hydro_cond = False):
    
    sample_idx = ds.get_iloc_from_date(date_max = sample_date)
    x = ds[sample_idx][0].unsqueeze(0)
    x_mask = ds[sample_idx][4].unsqueeze(0)

    w = [ds[sample_idx][2].unsqueeze(0).to(device),
        ds.weather_coords_dtm.unsqueeze(0).to(device)]
    
    lat_point = int(lon_point * 0.75)
    total_cpoint = lat_point * lon_point
    x_cpoint = x.expand(total_cpoint,-1,-1).to(device)
    x_mask_cpoint = x_mask.expand(total_cpoint,-1).to(device)
                    
    z_cpoint = ds.control_points_generator(mode = "even",
                                    num_lon_point = lon_point,
                                    num_lat_point = lat_point)

                    
    z_cpoint = torch.tensor(z_cpoint).to(torch.float32).to(device)
    w_cpoint = [w[0].expand(total_cpoint,-1,-1,-1,-1),
            w[1].expand(total_cpoint,-1,-1,-1)]
    
    if hydro_cond is True:            
        y_hat_cpoint, hc_hat = model(x_cpoint, z_cpoint, w_cpoint, x_mask_cpoint, hc_out = True)
    else:
        y_hat_cpoint = model(x_cpoint, z_cpoint, w_cpoint, x_mask_cpoint)
    
    # Adapt dtm resolution
    
    dtm_denorm = (ds.dtm_roi *ds.norm_factors["dtm_std"]) + ds.norm_factors["dtm_mean"]


    new_width = lon_point #int(dtm_roi.rio.width * downscale_factor)
    new_height = lat_point #int(dtm_roi.rio.height * downscale_factor)

    dtm_denorm_downsampled = dtm_denorm.rio.reproject(
        dtm_denorm.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )
    
    # de-normalize coords
    z_cpoint[:,0] = (z_cpoint[:,0]*ds.norm_factors["lat_std"]) +  ds.norm_factors["lat_mean"]
    z_cpoint[:,1] = (z_cpoint[:,1]*ds.norm_factors["lon_std"]) + ds.norm_factors["lon_mean"]
    
    # Xarray Creation
    
    h_pred = y_hat_cpoint.reshape(lat_point,lon_point,180).detach().cpu().numpy()
    h_pred = (h_pred* ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]

    sample_h = xarray.DataArray(data = h_pred,
                                coords = dict(
                                            lon=("lon", z_cpoint[:,1].unique().detach().cpu().numpy()),
                                            lat=("lat", z_cpoint[:,0].unique().detach().cpu().numpy()[::-1]),
                                            time=np.arange(180)),
                                dims = ["lat", "lon", "time"]
                                )

    wtd_pred = []
    for timestep in range(180):
        wtd_pred.append(dtm_denorm_downsampled.values.squeeze() - sample_h[:,:,timestep].values)

    wtd_pred = np.stack(wtd_pred, axis = -1)
    sample_wtd = xarray.DataArray(data = wtd_pred,
                               coords = dict(
                                        lon=("lon", z_cpoint[:,1].unique().detach().cpu().numpy()),
                                        lat=("lat", z_cpoint[:,0].unique().detach().cpu().numpy()[::-1]),
                                        time=np.arange(180)),
                               dims = ["lat", "lon", "time"]
                               )
    
    if hydro_cond is True:
        hc_hat = hc_hat.reshape(lat_point,lon_point,2).detach().cpu().numpy()
        
        sample_hc_lat = xarray.DataArray(data = hc_hat[:,:,0],
                                coords = dict(
                                            lon=("lon", z_cpoint[:,1].unique().detach().cpu().numpy()),
                                            lat=("lat", z_cpoint[:,0].unique().detach().cpu().numpy()[::-1])),
                                dims = ["lat", "lon"],
                                name = "hc_lat",
                                )
        
        sample_hc_lon = xarray.DataArray(data = hc_hat[:,:,1],
                                coords = dict(
                                            lon=("lon", z_cpoint[:,1].unique().detach().cpu().numpy()),
                                            lat=("lat", z_cpoint[:,0].unique().detach().cpu().numpy()[::-1])),
                                dims = ["lat", "lon"],
                                name = "hc_lon",
                                )
        
        sample_hc = xarray.merge([sample_hc_lat, sample_hc_lon])
        return [sample_h, sample_wtd, dtm_denorm_downsampled, sample_hc]
    
    else:
        return [sample_h, sample_wtd, dtm_denorm_downsampled]


def plot_one_map_target(sample_h, sample_wtd, dtm_denorm_downsampled, 
                 sample_date, pred_timestep,
                 save_dir = None, 
                 print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,3, figsize = (15,4))
    fig.suptitle(f"t0: {sample_date} - Prediction Timestep {pred_timestep}")

    sample_h[:,:,pred_timestep].plot(ax = ax[0])
    ax[0].set_title("Piezometric head")


    sample_wtd[:,:,pred_timestep].plot(ax = ax[1], vmin = sample_wtd.min().values,
                            vmax = sample_wtd.max().values)
    ax[1].set_title("WTD")


    dtm_denorm_downsampled.plot(ax = ax[2])
    ax[2].set_title("resampled DTM")

    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
        
    else:
        return fig 
    
def plot_one_map_hc(sample_hc,
                 save_dir = None, 
                 print_plot = False):
    
    ## Plot the maps
    
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    fig.suptitle(f"Hydraulic Condictivity")

    sample_hc["hc_lat"].plot(ax = ax[0])
    ax[0].set_title("HC - Lat")
    
    sample_hc["hc_lon"].plot(ax = ax[1])
    ax[1].set_title("HC - Lon")

    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
        
    else:
        return fig 
    
    
def plot_series_and_maps(ds, model, device, dates_list, tsteps_list, hydro_cond = False):

        for date in dates_list:
                    # Time Series  
                    wandb.log({f"pred_series_{date} - R":wandb.Image(plot_one_series(ds = ds,
                                                                date_t0 = np.datetime64(date),
                                                                sensor = 6,
                                                                model = model,
                                                                device = device,
                                                                print_plot = False))})
        
                    wandb.log({f"pred_series_{date} - V":wandb.Image(plot_one_series(ds = ds,
                                                                date_t0 = np.datetime64(date),
                                                                sensor = 15,
                                                                model = model,
                                                                device = device,
                                                                print_plot = False))})
                    
                    # Maps
                    if hydro_cond is True:
                        sample_h, sample_wtd, dtm_denorm_downsampled, sample_hc = predict_map_points(ds, lon_point = 40, 
                                                                                sample_date = date,
                                                                                model = model, device = device, 
                                                                                hydro_cond = True)
                        
                        wandb.log({f"hydraulic_conductivity_{date}":wandb.Image(plot_one_map_hc(sample_hc,
                                                                                    save_dir = None, 
                                                                                    print_plot = False))})
                    else:
                        sample_h, sample_wtd, dtm_denorm_downsampled = predict_map_points(ds, lon_point = 40, 
                                                                        sample_date = date,
                                                                        model = model, device = device)
                        
                    for tstep in tsteps_list:
        
                            wandb.log({f"pred_map_{date}-t{tstep}":wandb.Image(plot_one_map_target(sample_h, sample_wtd, dtm_denorm_downsampled, 
                                                                                date, pred_timestep = tstep,
                                                                                save_dir = None, 
                                                                                print_plot = False))})
                            
                    


################
##### 2-D ######
################

def plot_random_station_time_series(y, y_hat, i, save_dir = None, model_name = None, title = None, mode = "training",
                                    print_plot = False, wandb_log = True):

    pz_h_mask = y[0,1,0,:,:]
    avail_idxs = np.argwhere(pz_h_mask)
    idx_stat = random.randint(0, avail_idxs.shape[1]-1)
    coords_station = avail_idxs[:,idx_stat]

    x_y_plot = np.argwhere(y[0,1,:,coords_station[0],coords_station[1]])[0]
    y_plot = y[0,0,x_y_plot,coords_station[0],coords_station[1]]
    y_hat_plot = y_hat[0,0,:,coords_station[0],coords_station[1]]

    fig, ax = plt.subplots()
    fig.suptitle("Loss vs iterations")
    ax.plot(y_hat_plot, label = "predicted")
    ax.plot(x_y_plot, y_plot, label = "true", marker = "o")
    ax.legend()

    if title is not None:
        ax.set_title(title)
        
    if save_dir and model_name:
        plt.savefig(f"{save_dir}/timeseries_{model_name}_ep{i}_{mode}_{coords_station[0]}_{coords_station[1]}.png", bbox_inches = 'tight') # dpi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
        
    if wandb_log is True:
        wandb.log({
                    f"{mode}_timeseries_prediction" :  wandb.Image(fig,
                                                                caption=f"Prediction series ({coords_station[0]},{coords_station[1]}) ep{i} {mode}")
                })
        

def plot_2d_prediction(Y_hat, i, timestep, save_dir = None, model_name = None, mode = "training",
                       print_plot = False, wandb_log = True):
    
    fig, ax = plt.subplots(figsize = (10,10))
    fig.suptitle(f"Prediction t={timestep}")
    image = ax.imshow(Y_hat[0,0,timestep,:,:])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(image, cax = cax)
    
    if save_dir and model_name:
        plt.savefig(f"{save_dir}/image_t{timestep}_{i}_{model_name}_{mode}.png", bbox_inches = 'tight') #d pi = 400, transparent = True
    
    if print_plot is True:
        plt.tight_layout()
    
    if wandb_log is True:
        wandb.log({
            f"{mode}_image_prediction" :  wandb.Image(fig, caption=f"Prediction image ep{i} t{timestep} {mode}")
        })    

################
#### COMMON ####
################

                            
def plot_model_graph(file_path, file_name, model, sample_input, device, depth = 1):
    
    model_graph = draw_graph(model, input_data=sample_input, device=device, depth = depth)
    model_graph.visual_graph.render(format='png', filename = file_name, directory= f"{file_path}/")
    model_arch = wandb.Image(f"{file_path}/{file_name}.png", caption="model's architecture")
    
    return model_arch
            