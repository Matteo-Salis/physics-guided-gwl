from operator import itemgetter
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray

import rioxarray
import fiona
from rasterio.enums import Resampling
from geocube.api.core import make_geocube

import torch
from torch.utils.data import Dataset
import torch.nn as nn

import torch
from torch.utils.data import Dataset
import torch.nn as nn

class DiscreteDataset(Dataset):
    """..."""

    def __init__(self, dict_files, transform=None):
        """
        Args:
            dict_files (string): Path to the .nc file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dict_files = dict_files
        self.timesteps = self.dict_files["timesteps"]

        self.loading_weather()

        self.loading_rasterized_wtd()

        self.transform = transform

    def loading_weather(self):
        self.weather_xr = xarray.open_dataset(self.dict_files["weather_nc_path"])
        self.weather_xr = self.weather_xr.rio.write_crs("epsg:4326")

    def loading_rasterized_wtd(self):
        wtd_df = pd.read_csv(self.dict_files["wtd_csv_path"], dtype= {"sensor_id": "str"})
        wtd_df = wtd_df.astype({"date":'datetime64[ns]'})

        wtd_names  = gpd.read_file(self.dict_files["wtd_shp"], engine='fiona')
        wtd_names = wtd_names.to_crs('epsg:4326')

        dtm_roi = rioxarray.open_rasterio(self.dict_files["dtm_nc"], engine='fiona')
        dtm_roi = dtm_roi.rio.write_crs("epsg:4326")

        all_dates = wtd_df["date"].unique()
        sensor_id_list = wtd_df["sensor_id"].unique()
        
        # add coordinates to wtd values
        sensor_coord_x_list = []
        sensor_coord_y_list = []
        for sensor in sensor_id_list:
            coord_x = wtd_names.loc[wtd_names["CODICE_PUN"] == sensor].geometry.x.values[0]
            coord_y = wtd_names.loc[wtd_names["CODICE_PUN"] == sensor].geometry.y.values[0]
            sensor_coord_x_list.append(coord_x)
            sensor_coord_y_list.append(coord_y)

        from_id_to_coord_x_dict = {sensor_id_list[i]: sensor_coord_x_list[i] for i in range(len(sensor_coord_x_list))}
        from_id_to_coord_y_dict = {sensor_id_list[i]: sensor_coord_y_list[i] for i in range(len(sensor_coord_y_list))}

        queries = list(wtd_df["sensor_id"].values)
        coordinates_x = itemgetter(*queries)(from_id_to_coord_x_dict)
        coordinates_y = itemgetter(*queries)(from_id_to_coord_y_dict)

        wtd_df["x"] = coordinates_x
        wtd_df["y"] = coordinates_y

        # downscaling dtm
        downscale_factor = 0.1
        new_width = int(dtm_roi.rio.width * downscale_factor)
        new_height = int(dtm_roi.rio.height * downscale_factor)

        self.dtm_roi_downsampled = dtm_roi.rio.reproject(
            dtm_roi.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )

        wtd_data_geop = gpd.GeoDataFrame(
        wtd_df, geometry=gpd.points_from_xy(wtd_df["x"], 
                                            wtd_df["y"]), crs="EPSG:4326")
        wtd_data_geop = wtd_data_geop[["date","wtd","geometry"]]
        
        rasterized_ds_list = []
        for date_idx in range(len(all_dates)):
            
            vector_ds = wtd_data_geop.loc[wtd_data_geop["date"] == all_dates[date_idx],:]

            rasterized_ds = make_geocube(vector_data=vector_ds,
                                        measurements=['wtd'],
                                        output_crs="epsg:4326",
                                        resolution=(self.dtm_roi_downsampled.rio.transform().a, self.dtm_roi_downsampled.rio.transform().e))
            
            rasterized_ds_list.append(rasterized_ds)

        self.wtd_data_raserized = xarray.concat(rasterized_ds_list, dim = "time")
        self.wtd_data_raserized = self.wtd_data_raserized.assign_coords({"time": all_dates})

    def __len__(self):
        return len(self.wtd_data_raserized["wtd"]) - self.timesteps
    
    def __getitem__(self, idx):
        """"
        Input: (wtd_t, weather_t-T,t)
        Output: (wtd_t+1:t+T+1)        
        """
        if idx < 0:
            idx = self.__len__() + idx

        input_wtd = self.wtd_data_raserized["wtd"][idx].to_numpy()
        input_wtd_mask = np.ones_like(input_wtd) - np.isnan(input_wtd)
        input_wtd = np.stack((input_wtd, input_wtd_mask), axis=-1)

        d1 = self.wtd_data_raserized["time"][idx+1]
        d2 = self.wtd_data_raserized["time"][idx+self.timesteps]
        input_weather = self.weather_xr.sel(time=slice(d1, d2)).to_array().values

        output_wtd = self.wtd_data_raserized["wtd"][idx+1:idx+1+self.timesteps].to_numpy()
        output_wtd_mask = np.ones_like(output_wtd) - np.isnan(output_wtd)
        output_wtd = np.stack((output_wtd, output_wtd_mask), axis=-1)

        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(input_wtd), torch.from_numpy(input_weather), torch.from_numpy(output_wtd)

if __name__ == "__main__":
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/data.json') as f:
        dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

    ds = DiscreteDataset(dict_files)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")

    x,y,z = ds[-1]
    print(f"Sizes of ds[-1]: {x.shape} - {y.shape} - {z.shape}")



