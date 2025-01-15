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


class ContinuousDataset(Dataset):
    """Weather and WTD Dataset for the continuous case model"""

    def __init__(self, dict_files):
        """
        Args:
            dict_files (string): Path to the .nc file.
            transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        
        # Attributes init
        self.dict_files = dict_files
        self.timesteps = self.dict_files["timesteps"]

        # Meteorological data loading 
        self.loading_weather()
        
        # Digital Terrain Model data loading
        self.loading_dtm()
        
        # Water Table Depth data loading 
        self.loading_point_wtd(fill_value = dict_files["fill_value"])
        
        if dict_files["piezo_head"] is True:
            self.compute_piezo_head()
            self.target = "h"
        else:
            self.target = "wtd"
        
        if dict_files["normalization"] is True:
            
            self.normalize(date_max = np.datetime64(dict_files["date_max_norm"]))

        # Transform       
        self.transform = dict_files["transform"]
        
        
    def compute_norm_factors(self, date_max = np.datetime64("2020-01-01"), verbose = True, dict_out = False):
        
        subset_wtd_df = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= date_max,
                                                       :]] #
        subset_wtd_df = subset_wtd_df.loc[subset_wtd_df["nan_mask"] == True, :] #compute only for not nan values
        subset_weather_xr = self.weather_xr.sel(time = slice(date_max)) #slice include extremes
        
        target_mean = subset_wtd_df[self.target].mean()
        target_std = subset_wtd_df[self.target].std()
        dtm_mean = self.dtm_roi.mean()
        dtm_std = self.dtm_roi.std()
        lat_mean = self.weather_coords.mean(axis=(0,1))[0]
        lat_std = self.weather_coords.std(axis=(0,1))[0]
        lon_mean = self.weather_coords.mean(axis=(0,1))[1]
        lon_std = self.weather_coords.std(axis=(0,1))[1]
        weather_mean = subset_weather_xr.mean()
        weather_std = subset_weather_xr.std()
        
        self.norm_factors = {"target_mean": target_mean,
                            "target_std": target_std,
                            "dtm_mean": dtm_mean,
                            "dtm_std": dtm_std,
                            "lat_mean": lat_mean,
                            "lat_std": lat_std,
                            "lon_mean": lon_mean,
                            "lon_std": lon_std,
                            "weather_mean": weather_mean,
                            "weather_std": weather_std}
        
        if verbose is True:
            print("Norm factors:")
            print(self.norm_factors)
            print(f"Max date norm: {date_max}")
            
        if dict_out is True:
            return self.norm_factors
        
    def normalize(self, norm_factors = None, date_max = np.datetime64("2020-01-01")):
        if norm_factors is None:
            self.compute_norm_factors(date_max = date_max)
            ## compute norm factors by default
            
        else:
            ## if provided 
            self.norm_factors = norm_factors
        
        # Normalizations
        self.wtd_df[self.target] = (self.wtd_df[self.target] - self.norm_factors["target_mean"])/self.norm_factors["target_std"]
            
        self.wtd_df["lat"] = (self.wtd_df["lat"] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.wtd_df["lon"] = (self.wtd_df["lon"] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
            
        self.dtm_roi = (self.dtm_roi - self.norm_factors["dtm_mean"])/self.norm_factors["dtm_std"]
        self.wtd_df["height"] = (self.wtd_df["height"] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        self.weather_dtm = (self.weather_dtm - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
            
        self.weather_coords[:,:,0] = (self.weather_coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.weather_coords[:,:,1] = (self.weather_coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
            
        self.weather_xr = (self.weather_xr - self.norm_factors["weather_mean"])/self.norm_factors["weather_std"]       
            
        
    def loading_dtm(self):
        self.dtm_roi = rioxarray.open_rasterio(self.dict_files["dtm_nc"],
                                               engine='fiona')
        self.dtm_roi = self.dtm_roi.rio.write_crs("epsg:4326")
        
            
    def loading_weather(self):
        self.weather_xr = xarray.open_dataset(self.dict_files["weather_nc_path"])
        self.weather_xr = self.weather_xr.rio.write_crs("epsg:4326")
        
        # Compute coord matrix
        lat_matrix = np.vstack([self.weather_xr.lat.values for i in range(len(self.weather_xr.lon.values))]).transpose()
        lon_matrix = np.vstack([self.weather_xr.lon.values for i in range(len(self.weather_xr.lat.values))])
        
        self.weather_coords = np.stack([lat_matrix,lon_matrix], axis = -1)
        
        self.weather_dtm = rioxarray.open_rasterio(self.dict_files["weather_dtm"],
                                               engine='fiona')
        
        self.weather_dtm = self.weather_dtm.values
        self.weather_dtm = np.moveaxis(self.weather_dtm, 0,-1)

    def loading_point_wtd(self, fill_value = None):
        
        # Water Table Depth data loading
        self.wtd_df = pd.read_csv(self.dict_files["wtd_csv_path"], 
                                    dtype= {"sensor_id": "str"})
        self.wtd_df = self.wtd_df.astype({"date":'datetime64[ns]'})

        # Water Table Depth Sensors shapefile loading: 
        self.wtd_names = gpd.read_file(self.dict_files["wtd_shp"],
                                             engine='fiona')
        self.wtd_names = self.wtd_names.to_crs('epsg:4326')

        # Define attributes about dates and coordinates
        self.dates = self.wtd_df["date"].unique()
        self.sensor_id_list = self.wtd_df["sensor_id"].unique()
        # use same sensors order
        self.wtd_names["sensor_id"] = pd.Categorical(self.wtd_names["sensor_id"], ordered=True, categories=self.sensor_id_list)
        self.wtd_names = self.wtd_names.sort_values('sensor_id')
        self.wtd_names = self.wtd_names.reset_index(drop=True)
        
        # Find sensors' heights 
        dtm_values = [self.dtm_roi.sel(x = self.wtd_names.geometry.x.values[i],
                             y = self.wtd_names.geometry.y.values[i],
                             method = "nearest").values.squeeze() for i in range(len(self.sensor_id_list))]
                                
        self.wtd_names["height"] = np.array(dtm_values).squeeze() #add dtm values in the geopandas
        
        ### Merge csv and shp into a joint spatio temporal representation
        sensor_coord_x_list = []
        sensor_coord_y_list = []
        sensor_height = []

        # Retrieve coordinates from id codes
        for sensor in self.sensor_id_list:
            coord_x = self.wtd_names.loc[self.wtd_names["sensor_id"] == sensor].geometry.x.values[0]
            coord_y = self.wtd_names.loc[self.wtd_names["sensor_id"] == sensor].geometry.y.values[0]
            height = self.wtd_names["height"].loc[self.wtd_names["sensor_id"] == sensor].values[0]
            sensor_coord_x_list.append(coord_x)
            sensor_coord_y_list.append(coord_y)
            sensor_height.append(height)
            
        # Buil a dictionary of coordinates and id codes
        from_id_to_coord_x_dict = {self.sensor_id_list[i]: sensor_coord_x_list[i] for i in range(len(sensor_coord_x_list))}
        from_id_to_coord_y_dict = {self.sensor_id_list[i]: sensor_coord_y_list[i] for i in range(len(sensor_coord_y_list))}
        from_id_height_dict = {self.sensor_id_list[i]: sensor_height[i] for i in range(len(sensor_height))}

        # Map id codes to coordinates for all rows in the original ds
        queries = list(self.wtd_df["sensor_id"].values)
        coordinates_x = itemgetter(*queries)(from_id_to_coord_x_dict)
        coordinates_y = itemgetter(*queries)(from_id_to_coord_y_dict)
        heights = itemgetter(*queries)(from_id_height_dict)

        # insert new columns containing coordinates
        self.wtd_df["lon"] = coordinates_x
        self.wtd_df["lat"] = coordinates_y
        self.wtd_df["height"] = heights
        
        self.wtd_df = self.wtd_df.set_index(["date","sensor_id"])
        
        # Subset wtd data truncating the last `timestep` instances
        last_date = self.dates.max() - np.timedelta64(self.timesteps, 'D')
        self.input_dates = self.dates[self.dates <= last_date]
        
        # Create nan-mask
        self.wtd_df["nan_mask"] = ~self.wtd_df["wtd"].isna()
        
        if fill_value:
            #self.wtd_df["wtd"] = self.wtd_df["wtd"].fillna(fill_value)
            self.fill_value = fill_value
        else:
            self.fill_value = 0
        
    def compute_piezo_head(self):
        self.wtd_df["h"] = self.wtd_df["height"] - self.wtd_df["wtd"]
        
    def get_iloc_from_date(self, date_max):
        """
        return iloc of last sensor before date_max
        """
        row_num = self.wtd_df.index.get_loc(self.wtd_df[self.wtd_df.index.get_level_values(0) < date_max].iloc[-1].name)
        return row_num
        
    def __len__(self):
        data = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= self.input_dates.max(),
                                                       :]]
        return len(data)
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
        
        # Retrieve date and coords for idx instance
        start_date = self.wtd_df.iloc[idx, :].name[0]
        sample_lat = self.wtd_df["lat"].iloc[idx]
        sample_lon = self.wtd_df["lon"].iloc[idx]
        sample_dtm = self.wtd_df["height"].iloc[idx]  
        
        end_date = start_date + np.timedelta64(self.timesteps, "D")
        
        # Initial state WTD (t0) data
        target_t0 = self.wtd_df[[self.target, "nan_mask", "lat", "lon", "height"]].loc[self.wtd_df.index.get_level_values(0) == start_date]
        target_t0_values = target_t0[self.target].values
        target_t0_mask = target_t0["nan_mask"].values
        target_t0_lat = target_t0["lat"].values
        target_t0_lon = target_t0["lon"].values
        target_t0_dtm = target_t0["height"].values
        
        X = [torch.from_numpy(target_t0_lat).to(torch.float32),
             torch.from_numpy(target_t0_lon).to(torch.float32),
             torch.from_numpy(target_t0_dtm).to(torch.float32),
             torch.from_numpy(target_t0_values).to(torch.float32).nan_to_num(self.fill_value)
             ]
        X = torch.stack(X, dim = -1)
        
        Z = [torch.tensor(sample_lat).reshape(1).to(torch.float32),
             torch.tensor(sample_lon).reshape(1).to(torch.float32),
             torch.tensor(sample_dtm).reshape(1).to(torch.float32)]
        
        Z = torch.stack(Z, dim = -1).squeeze()
        
        # Retrieve weather data
        weather_video = self.weather_xr.sel(time = slice(start_date + np.timedelta64(1, "D"),
                                                    end_date)) #slice include extremes
        weather_video = weather_video.to_array().values
        W = torch.from_numpy(weather_video).to(torch.float32)
        
        # Retrieve wtd values from t0+1 to T for the idx instance sensor
        target_t1_T = self.wtd_df[[self.target, "nan_mask"]].loc[(self.wtd_df.index.get_level_values(0) > start_date) &
                                          (self.wtd_df.index.get_level_values(0) <= end_date)  & 
                                          (self.wtd_df["lat"] == sample_lat)&
                                          (self.wtd_df["lon"] == sample_lon)]
        
        target_t1_T_values =  target_t1_T[self.target].values
        target_t1_T_mask =  target_t1_T["nan_mask"].values        
        
        Y = torch.from_numpy(target_t1_T_values).to(torch.float32)
        
        X_mask = torch.from_numpy(target_t0_mask).to(torch.bool)
                 
        Y_mask = torch.from_numpy(target_t1_T_mask).to(torch.bool)
        
        if self.transform:
            sample = self.transform(sample)
        
        return [X, Z, W, Y, X_mask, Y_mask]
    
    def get_weather_dtm(self):
        return torch.from_numpy(self.weather_dtm).to(torch.float32)
        
    def get_weather_coords(self):
        return torch.from_numpy(self.weather_coords).to(torch.float32) 
    
     
    
if __name__ == "__main__":
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

    ds = ContinuousDataset(dict_files)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")



