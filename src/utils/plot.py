import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import torch
import wandb
import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

from rasterio.enums import Resampling

from torchview import draw_graph
from functools import partial

################
##### 1-D ######
################

def predict_series_points(ds, date_t0, sensor_number, model, device, teacher_training = False):

    sample_idx = ds.get_iloc_from_date(date_max = date_t0) + 1

    x, z, w_values, y, x_mask, y_mask = ds[sample_idx + sensor_number]

    x = x.to(device)
    x_mask = x_mask.to(device)
    z = z.to(device)
    w = [w_values.to(device), ds.weather_coords_dtm.to(device)]
    y = y.to(device)
    y_mask = y_mask.to(device)

    if teacher_training is True:
        y_hat = model(x, z, w, x_mask, (y.unsqueeze(0), y_mask.unsqueeze(0)))
    else:
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
                    generate_points = True, model = None, device = None,
                    teacher_training = False):
    
    """
    if generate_point = True provide the sensor number, otherwise the sensor_id
    """
    
    if generate_points is True:
        df_prediction, sensor_id = predict_series_points(ds, date_t0, sensor, model, device, teacher_training)

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
    
    
def plot_series_and_maps(ds, model, device, dates_list, tsteps_list, hydro_cond = False, teacher_training = False):

        for date in dates_list:
                    # Time Series  
                    wandb.log({f"pred_series_{date} - R":wandb.Image(plot_one_series(ds = ds,
                                                                date_t0 = np.datetime64(date),
                                                                sensor = 6,
                                                                model = model,
                                                                device = device,
                                                                print_plot = False,
                                                                teacher_training = teacher_training))})
        
                    wandb.log({f"pred_series_{date} - V":wandb.Image(plot_one_series(ds = ds,
                                                                date_t0 = np.datetime64(date),
                                                                sensor = 15,
                                                                model = model,
                                                                device = device,
                                                                print_plot = False,
                                                                teacher_training = teacher_training))})
                    
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
##### 2d ######
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
        
        
########################
###### SparseData ######
########################
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

    coords[:,:,0] = (coords[:,:,0] - dataset.norm_factors["lat_mean"])/dataset.norm_factors["lat_std"]
    coords[:,:,1] = (coords[:,:,1] - dataset.norm_factors["lon_mean"])/dataset.norm_factors["lon_std"]

    Z = np.concat([coords, np.moveaxis(dtm_grid, 0, -1)], axis=-1)

    Z = Z.reshape(Z.shape[0]*Z.shape[1], Z.shape[2])
    
    return Z

def compute_predictions(start_date, twindow, dataset, model, device, Z_grid = None, eval = True):
    
    start_date_input = start_date
    start_date_output = start_date + np.timedelta64(1, dataset.config["frequency"])
    
    end_date_input = start_date_input + np.timedelta64(twindow-1, dataset.config["frequency"])
    end_date_output = start_date_output + np.timedelta64(twindow-1, dataset.config["frequency"])
    
    if eval is True:
        model.eval()                                                    
        X, X_mask = dataset.get_target_data(start_date_input, start_date_input)    
        X = X.squeeze()
        X_mask = X_mask.squeeze()
        teacher_forcing = False
        
    else:
        X, X_mask = dataset.get_target_data(start_date_input, end_date_input)
        model.train()
        teacher_forcing = True
    
    if Z_grid is None:
        Z = torch.from_numpy(dataset.sparse_target_coords).to(torch.float32)
    else:
        Z = torch.from_numpy(Z_grid).to(torch.float32)
        
        
        
    W = dataset.get_weather_video(start_date_output,
                                  end_date_output)
    
    Y, _ = dataset.get_target_data(start_date_output,
                                   end_date_output,
                                   get_coord=False,
                                   fill_na = False)
    
    
    
    Y_hat = model(X = X.unsqueeze(0).to(device),
                    Z = Z.unsqueeze(0).to(device),
                    W = [W[0].unsqueeze(0).to(device), W[1].unsqueeze(0).to(device)],
                    X_mask = X_mask.unsqueeze(0).to(device),
                    teacher_forcing = teacher_forcing)
    
    return [Y.detach().cpu(),
            Y_hat.detach().cpu()]
    
    
def build_ds_from_pred(y_hat, start_date, twindow, freq, sensor_names):
    
    end_date = start_date + np.timedelta64(twindow - 1, freq)
    pd_ds = pd.DataFrame(data = y_hat,
                         index = pd.date_range(start_date, end_date, freq = freq),
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
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    if print_plot is True:
        plt.show()
        
    else:
        return fig
    
    


# plot_sensor_ts(sensor_ds)

def wandb_time_series(dataset, model, device,
                              start_dates_input, twindow,
                              sensors_to_plot, 
                              #timesteps_to_look,
                              eval_mode):
    
        
        for date in start_dates_input:
        
            Y_test, Y_hat_test = compute_predictions(start_date = np.datetime64(date),
                                                                            twindow = twindow,
                                                                            model = model,
                                                                            device = device,
                                                                            dataset = dataset,
                                                                            eval = eval_mode)
            
            
            # Y_hat_test_xr_denorm = build_xarray(data = Y_hat_test,
            #              dataset = dataset,
            #              start_date = date,
            #              twindow = twindow)
            
            #WTD_hat_test_xr_denorm = dataset.target_rasterized_dtm.values - Y_hat_test_xr_denorm
            
            Y_hat_test_ds = build_ds_from_pred(Y_hat_test,
                                               np.datetime64(date) + np.timedelta64(1, dataset.config["frequency"]),
                                               twindow, dataset.config["frequency"], dataset.sensor_id_list)
            Y_test_ds = build_ds_from_pred(Y_test,
                                           np.datetime64(date) + np.timedelta64(1, dataset.config["frequency"]),
                                           twindow, dataset.config["frequency"], dataset.sensor_id_list)
            
            # Denormalization
            Y_hat_test_ds = (Y_hat_test_ds * dataset.norm_factors["target_std"]) + dataset.norm_factors["target_mean"]
            Y_test_ds = (Y_test_ds * dataset.norm_factors["target_std"]) + dataset.norm_factors["target_mean"]
            
            # Y_test_xr_denorm = build_xarray(data = Y_test,
            #              dataset = dataset,
            #              start_date = date,
            #              twindow = twindow)
            
            
            # for timestep in timesteps_to_look:
                    
            #     wandb.log({f"pred_map_{date}-t{timestep}":wandb.Image(plot_h_wtd_maps(Y_hat_test_xr_denorm,
            #                                                                             WTD_hat_test_xr_denorm,
            #                                                                             date, timestep,
            #                                                                             save_dir = None, 
            #                                                                             print_plot = False))})
                              
            for sensor_id in sensors_to_plot:
                
                #municipality, lat, lon = find_munic_lat_lon_sensor(dataset, sensor_id)
                # sensor_pred_ds = find_sensor_pred_in_xr(Y_test_xr_denorm, Y_hat_test_xr_denorm,
                #                                         lat = lat,
                #                                         lon = lon,
                #                                         )
                
                municipality = dataset.wtd_names["munic"].loc[dataset.wtd_names["sensor_id"] == sensor_id].values[0]
                
                wandb.log({f"{municipality}_ts_{date} -":wandb.Image(plot_time_series(
                                                                            Y_hat_test_ds[sensor_id], Y_test_ds[sensor_id],
                                                                            title = f"{sensor_id} - {municipality} - from {date}",
                                                                            save_dir = None,
                                                                            print_plot = False))})
        
        
################
#### 2D ########
################

def test_data_prediction(start_date, twindow, dataset, model, device, eval = True):
    
    if eval is True:
        model.eval()                                                    
        X, X_mask = dataset.get_icon_target_data(start_date, start_date)    
        X = X.squeeze()
        X_mask = X_mask.squeeze()
        teacher_forcing = False
        
    else:
        X, X_mask = dataset.get_icon_target_data(start_date, start_date + np.timedelta64(twindow-1, dataset.config["frequency"]))
        model.train()
        teacher_forcing = True
        
    Z = torch.from_numpy(dataset.target_rasterized_coords).to(torch.float32)
    W = dataset.get_weather_video(start_date, end_date = start_date + np.timedelta64(twindow, dataset.config["frequency"]))
    Y, _ = dataset.get_target_video(dataset.get_iloc_from_date(start_date), twindow)
    
    
    
    Y_hat_test = model(X = X.unsqueeze(0).to(device),
                    Z = Z.unsqueeze(0).to(device),
                    W = [W[0].unsqueeze(0).to(device), W[1].unsqueeze(0).to(device)],
                    X_mask = X_mask.unsqueeze(0).to(device),
                    teacher_forcing = teacher_forcing)
    
    return [Y.detach().cpu(),
            Y_hat_test.detach().cpu()]
    
def build_xarray(data, dataset, start_date = None, twindow = None, variable = "piezo_height"):
    
    if variable == "piezo_height":
        denorm_data = (data * dataset.norm_factors["target_std"]) + dataset.norm_factors["target_mean"]
        xr_ds = xarray.DataArray(data = denorm_data,
                                coords = dict(
                                            lat=("lat", dataset.wtd_data_raserized.y.values),
                                            lon=("lon", dataset.wtd_data_raserized.x.values),
                                            time=pd.date_range(np.datetime64(start_date) + np.timedelta64(1, dataset.config["frequency"]),
                                                            np.datetime64(start_date) + np.timedelta64(twindow, dataset.config["frequency"]),
                                                            freq = dataset.config["frequency"])),
                                dims = ["time","lat", "lon"]
                                )
        
    elif variable == "K_lat_lon":
        
        xr_ds = xarray.DataArray(data = data,
                                coords = dict(
                                            lat=("lat", dataset.wtd_data_raserized.y.values),
                                            lon=("lon", dataset.wtd_data_raserized.x.values),
                                            bands=["K_lat", "K_lon"]),
                                dims = ["bands","lat", "lon"]
                                )
        
        
    return xr_ds

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
    
    fig, ax = plt.subplots()
    ax.plot(sensor_ds, label = sensor_ds.columns, marker = "o", lw = 0.7, markersize = 2)
    ax.tick_params(axis='x', rotation=50)

    ax.set_title(title)

    ax.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    if print_plot is True:
        plt.show()
        
    else:
        return fig
    
    
def find_munic_lat_lon_sensor(dataset, sensor_id):
    
    municipality = dataset.wtd_names.loc[dataset.wtd_names["sensor_id"] == sensor_id, "munic"].values[0]
    lat = dataset.wtd_names.loc[dataset.wtd_names["sensor_id"] == sensor_id].geometry.y.values[0]
    lon = dataset.wtd_names.loc[dataset.wtd_names["sensor_id"] == sensor_id].geometry.x.values[0]
    
    return municipality, lat, lon

def find_sensor_pred_in_xr(true_xr, pred_xr, lat, lon):
    ts_pred = pd.DataFrame({"True": true_xr.sel(lon  = lon, lat = lat, method = "nearest"),
                       "Predicted": pred_xr.sel(lon  = lon, lat = lat, method = "nearest").values,
                       },
                       index = pred_xr.sel(lon  = lon, lat = lat, method = "nearest").time.values)
    
    return ts_pred

def plot_maps_and_time_series(dataset, model, device,
                              start_dates, twindow,
                              sensors_to_plot, 
                              timesteps_to_look,
                              eval_mode):
    
        
        for date in start_dates:
        
            Y_test, Y_hat_test = test_data_prediction(start_date = np.datetime64(date),
                                                                            twindow = twindow,
                                                                            model = model,
                                                                            device = device,
                                                                            dataset = dataset,
                                                                            eval = eval_mode)
            
            
            Y_hat_test_xr_denorm = build_xarray(data = Y_hat_test,
                         dataset = dataset,
                         start_date = date,
                         twindow = twindow)
            
            WTD_hat_test_xr_denorm = dataset.target_rasterized_dtm.values - Y_hat_test_xr_denorm
            
            Y_test_xr_denorm = build_xarray(data = Y_test,
                         dataset = dataset,
                         start_date = date,
                         twindow = twindow)
            
            
            for timestep in timesteps_to_look:
                    
                wandb.log({f"pred_map_{date}-t{timestep}":wandb.Image(plot_h_wtd_maps(Y_hat_test_xr_denorm,
                                                                                        WTD_hat_test_xr_denorm,
                                                                                        date, timestep,
                                                                                        save_dir = None, 
                                                                                        print_plot = False))})
                              
            for sensor_id in sensors_to_plot:
                
                municipality, lat, lon = find_munic_lat_lon_sensor(dataset, sensor_id)
                sensor_pred_ds = find_sensor_pred_in_xr(Y_test_xr_denorm, Y_hat_test_xr_denorm,
                                                        lat = lat,
                                                        lon = lon,
                                                        )
                
                wandb.log({f"{municipality}_ts_{date} -":wandb.Image(plot_sensor_ts(sensor_pred_ds,
                                                                            title = f"{sensor_id} - {municipality} - from {date}",
                                                                            save_dir = None,
                                                                            print_plot = False))})
                
                
                
                
###### GIF #######

def generate_gif_h_wtd(start_date, twindow,
                       sample_h, sample_wtd,
                       freq,
                       save_dir = None,
                       print_plot = False):

    fig, ax = plt.subplots(1,2, figsize = (10,4))

    ax[0].set_title("Piezometric head [m]")
    ax[1].set_title("WTD [m]")


    fig.suptitle(f"t0: {start_date} - Prediction Timestep {0}")
    piezo_image = sample_h[0,:,:].plot(ax = ax[0], animated=True,
                                                vmin = sample_h.min().values,
                                                vmax = sample_h.max().values,
                                                cmap = "Blues")
    wtd_image = sample_wtd[0,:,:].plot(ax = ax[1], animated=True, 
                                                vmin = sample_wtd.min().values,
                                                vmax = sample_wtd.max().values,
                                                cmap = "Greys")


    def update_h_wtd_maps(i):
        
        sample_date_i = np.datetime64(start_date) + np.timedelta64(i+1, freq)
        fig.suptitle(f"t0: {start_date} - Prediction Timestep {i}: {sample_date_i} ")
        
        ax[0].set_title("Piezometric head")
        ax[1].set_title("WTD")
        
        piezo_image.set_array(sample_h[i,:,:])
        wtd_image.set_array(sample_wtd[i,:,:])
        
        return (piezo_image, wtd_image)
        
        ## Plot the maps
        
        
    ani = animation.FuncAnimation(fig, update_h_wtd_maps, repeat=True, frames=twindow, interval=1)

    writer = animation.PillowWriter(fps=5,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    
    if save_dir:
        ani.save(f'{save_dir}.gif', writer=writer)
        
    if print_plot is True:
        plt.show()
                
        

################
#### COMMON ####
################

                            
def plot_model_graph(file_path, file_name, model, sample_input, device, depth = 1):
    
    model_graph = draw_graph(model, input_data=sample_input, device=device, depth = depth, mode = "train")
    model_graph.visual_graph.render(format='png', filename = file_name, directory= f"{file_path}/")
    model_arch = wandb.Image(f"{file_path}/{file_name}.png", caption="model's architecture")
    
    return model_arch
            