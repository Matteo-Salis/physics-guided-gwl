import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import torch
import wandb


import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from utils.pde_utils import * 

# def plot_predictions(x, y, y_hat, save_dir = None, title = None):
#     fig, ax = plt.subplots()
#     fig.suptitle("Loss vs iterations")
#     ax.plot(x, y_hat, label = "predicted")
#     ax.plot(x, y, label = "true")
#     ax.legend()
    
#     if title is not None:
#         ax.set_title(title)
        
#     if save_dir:
#         plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
#     return fig

def predict_series_points(ds, date_t0, sensor_number, model, device):
    
    weather_coords = ds.get_weather_coords()
    weather_dtm = ds.get_weather_dtm()
    weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)

    sample_idx = ds.get_iloc_from_date(date_max = date_t0) + 1

    x, z, w_values, y, x_mask, y_mask = ds[sample_idx + sensor_number]

    x = x.to(device)
    x_mask = x_mask.to(device)
    z = z.to(device)
    w = [w_values.to(device), weather_coords.to(device)]
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
        df_prediction = predict_series_points(ds, date_t0, sensor, model, device)[0]
        sensor_id = predict_series_points(ds, date_t0, sensor, model, device)[1]

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
                 model, device):
    
    sample_idx = ds.get_iloc_from_date(date_max = sample_date)
    x = ds[sample_idx][0].unsqueeze(0)
    x_mask = ds[sample_idx][4].unsqueeze(0)

    weather_coords = ds.get_weather_coords()
    weather_dtm = ds.get_weather_dtm()
    weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)
    weather_coords_batch = weather_coords.unsqueeze(0)

    w = [ds[sample_idx][2].unsqueeze(0).to(device),
        weather_coords_batch.to(device)]
    
    lat_point = int(lon_point * 0.75)
    total_cpoint = lat_point * lon_point
    x_cpoint = x.expand(total_cpoint,-1,-1).to(device)
    x_mask_cpoint = x_mask.expand(total_cpoint,-1).to(device)
                    
    z_cpoint = cpoint_generation(minX = ds.dtm_roi.x.min().values, maxX = ds.dtm_roi.x.max().values,
                                                minY = ds.dtm_roi.y.min().values, maxY = ds.dtm_roi.y.max().values,
                                                dtm = (ds.dtm_roi *ds.norm_factors["dtm_std"]) + ds.norm_factors["dtm_mean"],
                                                mode = "even",
                                                num_lon_point = lon_point,
                                                num_lat_point = lat_point)

    # normalization 
    z_cpoint[:,0] = (z_cpoint[:,0] - ds.norm_factors["lat_mean"])/ds.norm_factors["lat_std"]
    z_cpoint[:,1] = (z_cpoint[:,1] - ds.norm_factors["lon_mean"])/ds.norm_factors["lon_std"]
    z_cpoint[:,2] = (z_cpoint[:,2] - ds.norm_factors["dtm_mean"].values)/ds.norm_factors["dtm_std"].values
                    
    z_cpoint = torch.tensor(z_cpoint).to(torch.float32).to(device)
    w_cpoint = [w[0].expand(total_cpoint,-1,-1,-1,-1),
            w[1].expand(total_cpoint,-1,-1,-1)]
                
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
    
    return [sample_h, sample_wtd, dtm_denorm_downsampled]


def plot_one_map(sample_h, sample_wtd, dtm_denorm_downsampled, 
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
    
    
def train_test_plots(ds, model, device, dates_list, tsteps_list):

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
                    sample_h, sample_wtd, dtm_denorm_downsampled = predict_map_points(ds, lon_point = 40, 
                                sample_date = date,
                                model = model, device = device)
                    
                    for tstep in tsteps_list:
        
                            wandb.log({f"pred_map_{date}-t{tstep}":wandb.Image(plot_one_map(sample_h, sample_wtd, dtm_denorm_downsampled, 
                                        date, pred_timestep = tstep,
                                        save_dir = None, 
                                        print_plot = False))})