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

    def __init__(self, dict_files, #meteo_nc_path, wtd_csv_path, wtd_stations_shp_path,
                 transform = None):
        """
        Args:
            dict_files (string): Path to the .nc file.
            transform (callable, optional): Optional transform to be applied
                    on a sample.
                    
        Output: list of list: [X,Z,W,Y] 
                        X: list of tensors [wtd_t0_lat, wtd_t0_lon,
                            wtd_t0_dtm,
                            wtd_t0_values, wtd_t0_mask]      
                        Z: list of tensors [sample_lat, sample_lon, sample_dtm]
                        W: list of tensors [weather_video, self.weather_coords]
                        Y: list of tensors [wtd_t1_T_values,  wtd_t1_T_mask]
        """
        
        # Attributes init
        self.dict_files = dict_files
        self.timesteps = self.dict_files["timesteps"]

        # Meteorological data loading 
        self.loading_weather()
        
        # Digital Terrain Model data loading
        self.loading_dtm()
        
        # Water Table Depth data loading 
        self.loading_point_wtd()

        # Transform       
        self.transform = transform
        
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
        

    def loading_point_wtd(self):
        
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
        
        
        ### Merge csv and shp into a joint spatio temporal representation
        sensor_coord_x_list = []
        sensor_coord_y_list = []

        # Retrieve coordinates from id codes
        for sensor in self.sensor_id_list:
            coord_x = self.wtd_names.loc[self.wtd_names["sensor_id"] == sensor].geometry.x.values[0]
            coord_y = self.wtd_names.loc[self.wtd_names["sensor_id"] == sensor].geometry.y.values[0]
            sensor_coord_x_list.append(coord_x)
            sensor_coord_y_list.append(coord_y)

        # Buil a dictionary of coordinates and id codes
        from_id_to_coord_x_dict = {self.sensor_id_list[i]: sensor_coord_x_list[i] for i in range(len(sensor_coord_x_list))}
        from_id_to_coord_y_dict = {self.sensor_id_list[i]: sensor_coord_y_list[i] for i in range(len(sensor_coord_y_list))}

        # Map id codes to coordinates for all rows in the original ds
        queries = list(self.wtd_df["sensor_id"].values)
        coordinates_x = itemgetter(*queries)(from_id_to_coord_x_dict)
        coordinates_y = itemgetter(*queries)(from_id_to_coord_y_dict)

        # insert new columns containing coordinates
        self.wtd_df["x"] = coordinates_x
        self.wtd_df["y"] = coordinates_y
        
        self.wtd_df = self.wtd_df.set_index(["date","y","x"])
        
        # Truncate dataset wrt `timestep`
        
        # Subset wtd data truncating the last `timestep` instances
        last_date = self.dates.max() - np.timedelta64(self.timesteps, 'D')
        self.input_dates = self.dates[self.dates <= last_date]
        
    def __len__(self):
        data = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= self.input_dates.max(),
                                                       :,
                                                       :]]
        return len(data)
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
        
        # Retrieve Features
        ## Retrieve date and coords for idx instance
        start_date = self.wtd_df.iloc[idx, :].name[0]
        sample_lat = self.wtd_df.iloc[idx, :].name[1]
        sample_lon = self.wtd_df.iloc[idx, :].name[2]
        sample_dtm = self.dtm_roi.sel(x = sample_lon,
                                      y = sample_lat,
                                      method = "nearest").values  
        
        end_date = start_date + np.timedelta64(self.timesteps, "D")
        
        ## Initial state WTD (t0) data
        wtd_t0 = self.wtd_df["wtd"].loc[self.wtd_df.index.get_level_values(0) == start_date]
        wtd_t0_values = wtd_t0.values
        wtd_t0_lat = wtd_t0.index.get_level_values(1).values
        wtd_t0_lon = wtd_t0.index.get_level_values(2).values
        wtd_t0_dtm = np.array([self.dtm_roi.sel(x = wtd_t0_lon[sensor],
                                                y = wtd_t0_lat[sensor],
                                                method = "nearest") for sensor in range(len(wtd_t0_lat))])
        
        wtd_t0_mask = np.isnan(wtd_t0_values)
        
        X = [torch.from_numpy(wtd_t0_lat),
             torch.from_numpy(wtd_t0_lon),
             torch.from_numpy(wtd_t0_dtm),
             torch.from_numpy(wtd_t0_values),
             torch.from_numpy(wtd_t0_mask)]
        
        Z = [torch.tensor(sample_lat),
             torch.tensor(sample_lon),
             torch.tensor(sample_dtm)]
        
        ## Retrieve weather data
        weather_video = self.weather_xr.sel(time = slice(start_date + np.timedelta64(1, "D"),
                                                    end_date)) #slice include extremes
        weather_video = weather_video.to_array().values
        W = [torch.from_numpy(weather_video),
             torch.from_numpy(self.weather_coords)]
        
        # Retrieve Target
        # wtd values from t0+1 to T for the idx instance sensor
        wtd_t1_T = self.wtd_df["wtd"].loc[(self.wtd_df.index.get_level_values(0) > start_date) &
                                          (self.wtd_df.index.get_level_values(0) <= end_date)  & 
                                          (self.wtd_df.index.get_level_values(1) == sample_lat)&
                                          (self.wtd_df.index.get_level_values(2) == sample_lon)]
        
        wtd_t1_T_values =  wtd_t1_T.values
        wtd_t1_T_mask =  np.isnan(wtd_t1_T_values)
        
        Y = [torch.from_numpy(wtd_t1_T_values),
             torch.from_numpy(wtd_t1_T_mask)]
        
        if self.transform:
            sample = self.transform(sample)
        
        return [X, Z, W, Y]
    
    
if __name__ == "__main__":
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/data.json') as f:
        dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

    ds = ContinuousDataset(dict_files)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")

    x,z,w,y = ds[-1]
    print(f"len of ds[-1]: {len(x)} - {len(z)} - {len(w)} - {len(y)}")



