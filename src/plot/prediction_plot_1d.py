import numpy as np
import pandas as pd
import geopandas as gpd
import torch

import matplotlib.pyplot as plt

def plot_predictions(x, y, y_hat, save_dir = None, title = None):
    fig, ax = plt.subplots()
    fig.suptitle("Loss vs iterations")
    ax.plot(x, y_hat, label = "predicted")
    ax.plot(x, y, label = "true")
    ax.legend()
    
    if title is not None:
        ax.set_title(title)
        
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    
    return fig

def plot_one_instance(ds, date_t0, sensor_number, model, device, save_dir = None):

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

    prediction = pd.DataFrame({"prediction": y_hat.detach().cpu().numpy(),
                            "true": y.detach().cpu().numpy()}, index = dates )

    prediction["prediction"] = (prediction["prediction"] * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]
    prediction["true"] = (prediction["true"] * ds.norm_factors["target_std"]) + ds.norm_factors["target_mean"]

    fig, ax = plt.subplots()
    ax.plot(prediction, label = prediction.columns, marker = "o", lw = 0.7, markersize = 2)

    municipality = ds.wtd_names["munic"].loc[ds.wtd_names["sensor_id"] == sensor].values[0]
    ax.set_title(f"{sensor} - {municipality} - from {date_t0}")

    ax.legend()
    
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
        
    return fig

####### MAP

def cpoint_generation(minX, maxX, minY, maxY, dtm,
                      mode = "even",
                      num_lon_point = 100,
                      num_lat_point = 100):
    
    if mode == "even":
        
        # create one-dimensional arrays for x and y
        x = np.linspace(minX, maxX, num_lon_point)
        y = np.linspace(minY, maxY, num_lat_point)[::-1]
        # create the mesh based on these arrays
        X, Y = np.meshgrid(x, y)
        coords = np.stack([Y, X], axis = -1)
        
        
    dtm_xy = dtm.sel(x = x, y = y,
                     method = "nearest").values
    
    coords = np.concat([coords, np.moveaxis(dtm_xy, 0, -1)], axis=-1)
    coords = coords.reshape(coords.shape[0]*coords.shape[1], coords.shape[2])
        
    return coords