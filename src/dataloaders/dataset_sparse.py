from operator import itemgetter
import json
from functools import partial
import copy

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray

import rioxarray
import fiona
from rasterio.enums import Resampling
from geocube.api.core import make_geocube
from shapely.geometry import box

import torch
from torch.utils.data import Dataset
import torch.nn as nn

import warnings

class Dataset_Sparse(Dataset):
    """Weather and Groundwater Dataset for the continuous case model"""

    def __init__(self, config):
        """
        Args:
            config (string): Path to the .nc file.
            transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        
        # Attributes init
        self.config = config
        self.twindow = self.config["twindow"]

        # Meteorological data loading 
        print("    Loading weather data...", end = " ")
        self.loading_weather()
        print("Done!")
        
        # Digital Terrain Model data loading
        print("    Loading dtm...", end = " ")
        self.loading_dtm()
        print("Done!")
        
        # Water Table Depth data loading 
        print("    Loading underground water data...", end = " ")
        self.loading_point_wtd(fill_value = config["fill_value"])
        print("Done!")
        
        if config["piezo_head"] is True:
            self.compute_piezo_head()
            self.target = "h"
        else:
            self.target = "wtd"
            
        # Rasterizing groundwater dataset 
        # self.rasterize_sparse_measurements(new_dimensions = config["upsampling_dim"][1:])
        # around 2.5km
        
        if config["normalization"] is True:
            
            self.normalize(date_max = np.datetime64(config["date_max_norm"]))
        
        self.sparse_target_coords_object()
        
    # def create_numpy_objects(self):
    #     "create numpy for dataloader efficiency"
    #     self.target_rasterized_numpy = self.wtd_data_raserized.to_array().values.astype(np.float32)
        

        
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
            print("    Norm factors:")
            print(self.norm_factors)
            print(f"    Max date norm: {date_max}")
            
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
        #self.wtd_data_raserized[self.target] = (self.wtd_data_raserized[self.target] - self.norm_factors["target_mean"])/self.norm_factors["target_std"]
            
        self.dtm_roi = (self.dtm_roi - self.norm_factors["dtm_mean"])/self.norm_factors["dtm_std"]
        self.wtd_df["height"] = (self.wtd_df["height"] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        
        # self.target_rasterized_coords[:,:,0] = (self.target_rasterized_coords[:,:,0] - self.norm_factors["lat_mean"].values)/self.norm_factors["lat_std"].values
        # self.target_rasterized_coords[:,:,1] = (self.target_rasterized_coords[:,:,1] - self.norm_factors["lon_mean"].values)/self.norm_factors["lon_std"].values
        # self.target_rasterized_coords[:,:,2] = (self.target_rasterized_coords[:,:,2] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        
        norm_weather_var = list(self.weather_xr.keys())
        norm_weather_var = [var for var in norm_weather_var if var not in ["tmax","tmin"]]
        self.weather_xr[norm_weather_var] = (self.weather_xr[norm_weather_var] - self.norm_factors["weather_mean"][norm_weather_var])/self.norm_factors["weather_std"][norm_weather_var]
        
        self.weather_xr["tmin"] = (self.weather_xr["tmin"] - self.norm_factors["weather_mean"]["tmean"])/self.norm_factors["weather_std"]["tmean"]
        self.weather_xr["tmax"] = (self.weather_xr["tmax"] - self.norm_factors["weather_mean"]["tmean"])/self.norm_factors["weather_std"]["tmean"]
        
        self.weather_coords[:,:,0] = (self.weather_coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.weather_coords[:,:,1] = (self.weather_coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        self.weather_coords[:,:,2] = (self.weather_coords[:,:,2] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
            
        
    def loading_dtm(self):
        self.dtm_roi = rioxarray.open_rasterio(self.config["dtm_nc"],
                                               engine='fiona')
        self.dtm_roi = self.dtm_roi.rio.write_crs("epsg:4326")
        
    def coordinates_xr(self, xr, coord_name = "latlon"):
        
        """
        Compute matrix of coordinates (lat and lon) of a generic xarray. Usually raster has xy, while nc lon lat
        """
        if coord_name == "latlon":
            # Compute coord matrix
            lat_matrix = np.vstack([xr.lat.values for i in range(len(xr.lon.values))]).transpose()
            lon_matrix = np.vstack([xr.lon.values for i in range(len(xr.lat.values))])
        
        elif coord_name == "xy":
            lat_matrix = np.vstack([xr.y.values for i in range(len(xr.x.values))]).transpose()
            lon_matrix = np.vstack([xr.x.values for i in range(len(xr.y.values))])
            
        coords = np.stack([lat_matrix,lon_matrix], axis = -1)
        
        return coords
        
            
    def loading_weather(self):
        self.weather_xr = xarray.open_dataset(self.config["weather_nc_path"])
        
        self.weather_xr = self.weather_xr.resample(time=self.config["frequency"], label = "left").mean()
        
        self.weather_xr = self.weather_xr.rio.write_crs("epsg:4326")
        self.weather_xr = self.weather_xr[["prec", "tmax", "tmin", "tmean"]]
        
        self.weather_dtm = rioxarray.open_rasterio(self.config["weather_dtm"],
                                               engine='fiona')
        self.weather_coords = self.coordinates_xr(self.weather_dtm, coord_name = "xy")
        
        self.weather_coords = np.concat([self.weather_coords,
                                    np.moveaxis(self.weather_dtm.values, 0,-1)],
                                    axis=-1)
        
        if self.config["weather_get_coords"] is True:
            self.weather_get_coords = True
        else:
            self.weather_get_coords = False
        
    # def rasterize_sparse_measurements(self, new_dimensions):
    #     # downscaling dtm
        
    #     new_height, new_width = new_dimensions
    #     # new_width = round(self.dtm_roi.rio.width * downscale_factor)
    #     # new_height = round(self.dtm_roi.rio.height * downscale_factor)

    #     self.dtm_roi_downsampled = self.dtm_roi.rio.reproject(
    #         self.dtm_roi.rio.crs,
    #         shape=(new_height, new_width),
    #         resampling=Resampling.bilinear,
    #     )

    #     wtd_df_util = copy.deepcopy(self.wtd_df)
    #     wtd_df_util["date"] = self.wtd_df.index.get_level_values(0)
    #     wtd_data_geop = gpd.GeoDataFrame(
    #     wtd_df_util, geometry =gpd.points_from_xy(wtd_df_util["lon"], 
    #                                         wtd_df_util["lat"]), crs="EPSG:4326")
    #     wtd_data_geop = wtd_data_geop[["date",self.target,"geometry"]]

    #     self.minx = self.dtm_roi_downsampled.x.min()
    #     self.miny = self.dtm_roi_downsampled.y.min()
    #     self.maxx = self.dtm_roi_downsampled.x.max()
    #     self.maxy = self.dtm_roi_downsampled.y.max()
        
    #     print("Rasterizing groundwater dataframe...")
    #     rasterized_ds_list = []
    #     for date_idx in range(len(self.dates)):
            
    #         vector_ds = wtd_data_geop.loc[wtd_data_geop["date"] == self.dates[date_idx],:]

    #         rasterized_ds = make_geocube(vector_data=vector_ds,
    #                                     measurements=[self.target],
    #                                     output_crs="epsg:4326",
    #                                     # resolution= new_resolution,
    #                                     resolution=(self.dtm_roi_downsampled.rio.transform().a, round(self.dtm_roi_downsampled.rio.transform().e, 4)),
    #                                     # Global extent in degrees of longitude and latitude
    #                                     geom=box(minx=self.minx, miny=self.miny, maxx=self.maxx, maxy=self.maxy))
            
    #         rasterized_ds_list.append(rasterized_ds)
    #     print("Rasterization complete!")

    #     self.wtd_data_raserized = xarray.concat(rasterized_ds_list, dim = "time")
    #     self.wtd_data_raserized = self.wtd_data_raserized.assign_coords({"time": self.dates})

    #     self.wtd_data_raserized = self.wtd_data_raserized.reindex(y=list(reversed(self.wtd_data_raserized.y)))
    #     self.wtd_data_raserized = self.wtd_data_raserized.reindex(x=list(reversed(self.wtd_data_raserized.x)))

    #     mask = self.wtd_data_raserized.notnull()
    #     mask = mask.rename({self.target:'nan_mask'})
    #     self.wtd_data_raserized = self.wtd_data_raserized.assign(mask = mask["nan_mask"])
        
    #     self.target_rasterized_dtm = self.dtm_roi.sel(x = self.wtd_data_raserized.x,
    #                                                      y = self.wtd_data_raserized.y,
    #                                                      method = "nearest")
    #     self.target_rasterized_coords = self.coordinates_xr(self.target_rasterized_dtm, coord_name = "xy")
        
    #     self.target_rasterized_coords = np.concat([self.target_rasterized_coords,
    #                                 np.moveaxis(self.target_rasterized_dtm.values, 0,-1)],
    #                                 axis=-1)

    def loading_point_wtd(self, fill_value = None):
        
        # Water Table Depth data loading
        self.wtd_df = pd.read_csv(self.config["wtd_csv_path"], 
                                    dtype= {"sensor_id": "str"})
        self.wtd_df = self.wtd_df.astype({"date":'datetime64[ns]'})

        # Water Table Depth Sensors shapefile loading: 
        self.wtd_names = gpd.read_file(self.config["wtd_shp"],
                                             engine='fiona')
        self.wtd_names = self.wtd_names.to_crs('epsg:4326')
        
        # Subset Stations
        if self.config["discard_sensor_list"] is not None:
            # discard_sensor_list = ["00405910001",
            #                         "00119710001",
            #                         "00127210003",
            #                         "00117110001",
            #                         "00421710001",
            #                         "00121510001",
            #                         "00104810001",
            #                         "00127210005",
            #                         "00402910001",
            #                         "00403410001",
            #                         "00126010001",
            #                         "00105110001"]
            self.wtd_names = self.wtd_names.loc[~self.wtd_names["sensor_id"].isin(self.config["discard_sensor_list"]), :]
            self.wtd_df = self.wtd_df.loc[~self.wtd_df["sensor_id"].isin(self.config["discard_sensor_list"]), :]
        
        
                # sensor_list = ["00417910001",
                #                "00105910001",
                #                "00425010001",
                #                "00109010001",
                #                "00127210001"
                #                ]
                
                # self.wtd_names = self.wtd_names.loc[self.wtd_names["sensor_id"].isin(sensor_list), :]
                # self.wtd_df = self.wtd_df.loc[self.wtd_df["sensor_id"].isin(sensor_list), :]
                
                # self.wtd_df = self.wtd_df.set_index(["date", "sensor_id"])
                # self.wtd_df = self.wtd_df.groupby(level='date').filter(lambda group: not group.isna().all().all())
                # self.wtd_df = self.wtd_df.reset_index()
        # Resampling 
        
        if self.config["frequency"] != "D":
            self.wtd_df.sort_values(by='date', inplace = True)
            self.wtd_df = self.wtd_df.set_index(["date"])
            self.wtd_df = self.wtd_df.groupby([pd.Grouper(freq=self.config["frequency"], label = "left"), "sensor_id"]).mean()
            self.wtd_df = self.wtd_df.reset_index()
            
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
        
        # Subset wtd data truncating the last `twindow` instances
        last_date = self.dates.max() - np.timedelta64(self.twindow, self.config["frequency"])
        self.input_dates = self.dates[self.dates <= last_date]
        
        # Create nan-mask
        self.wtd_df["nan_mask"] = ~self.wtd_df["wtd"].isna()
        
        if fill_value:
            self.fill_value = fill_value
        else:
            self.fill_value = 0
            
    def sparse_target_coords_object(self):
        self.sparse_target_coords = self.wtd_df.loc[pd.IndexSlice[self.input_dates[0], :],
                                                    ["lat","lon","height"]]
        
        z_lat = self.sparse_target_coords["lat"].values
        z_lon = self.sparse_target_coords["lon"].values
        z_height = self.sparse_target_coords["height"].values
        
        self.sparse_target_coords = [z_lat,
                                    z_lon,
                                    z_height]
        
        self.sparse_target_coords = np.stack(self.sparse_target_coords, axis = -1).squeeze()
        
    def compute_piezo_head(self):
        self.wtd_df["h"] = self.wtd_df["height"] - self.wtd_df["wtd"]
        
    def get_iloc_from_date(self, date_max):
        """
        return iloc of last sensor before date_max
        """
        ds_iloc = np.argwhere(self.input_dates <= date_max).max()
        #row_num = self.wtd_df.index.get_loc(self.wtd_df[self.wtd_df.index.get_level_values(0) == date_max].iloc[-1].name)
        return ds_iloc
        
        # idx_subset = self.wtd_data_raserized.time.values[self.wtd_data_raserized.time.values < date_max]
        # if idx_subset.size > 0:
        #     max_idx = np.argmax(self.wtd_data_raserized.time.values[self.wtd_data_raserized.time.values < date_max])
        # else:
        #     max_idx = 0
        # return max_idx
        
    def __len__(self):
        #return len(self.wtd_data_raserized[self.target]) - self.twindow
        
        # data = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= self.input_dates.max(),
        #                                                :]]
        return len(self.input_dates)
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
        
        # Retrieve date and coords for idx instance
        start_date_input = np.datetime64(self.dates[idx])
        start_date_output = np.datetime64(self.dates[idx+1])
        
        end_date_input = start_date_input + np.timedelta64(self.twindow-1, self.config["frequency"])
        end_date_output = start_date_output + np.timedelta64(self.twindow-1, self.config["frequency"])
        
        Z = torch.from_numpy(self.sparse_target_coords).to(torch.float32) 
        
        # Initial state WTD (t0) data        
        X, X_mask = self.get_target_data(start_date_input, end_date_input, get_coord=True)
        
        # Retrieve weather data
        W = self.get_weather_video(start_date_output, end_date_output)
        
        # Retrieve wtd values from t0+1 to T for the idx instance sensor
        Y, Y_mask = self.get_target_data(start_date_output, end_date_output, get_coord=False)
        
        return [X, Z, W, Y, X_mask, Y_mask]
    
    def get_target_data(self, start_date, end_date, get_coord = True, fill_na = True):
        
        if get_coord is True:
            var_names = ["lat", "lon", "height", self.target, "nan_mask"]
        else:
            var_names = [self.target, "nan_mask"]
        
        target_subset_ds = self.wtd_df[var_names].loc[pd.IndexSlice[start_date:end_date, :]] #.loc[self.wtd_df.index.get_level_values(0) == date]
        timesteps = (end_date-start_date).astype(f'timedelta64[{self.config["frequency"]}]').astype(int) + 1
        
        target_tensor = []
        
        for var in var_names:
            
            target_data = torch.from_numpy(target_subset_ds[var].values.reshape((timesteps, len(self.sensor_id_list))))
            target_data = target_data.to(torch.float32) if var != "nan_mask" else target_data.to(torch.bool)
            target_tensor.append(target_data)
            
        # target_t0_values = target_t0[self.target].values.reshape((timesteps, len(self.sensor_id_list)))
        # target_t0_mask = target_t0["nan_mask"].values.reshape((timesteps, len(self.sensor_id_list)))
        # target_t0_lat = target_t0["lat"].values.reshape((timesteps, len(self.sensor_id_list)))
        # target_t0_lon = target_t0["lon"].values.reshape((timesteps, len(self.sensor_id_list)))
        # target_t0_dtm = target_t0["height"].values.reshape((timesteps, len(self.sensor_id_list)))
        
        target_mask = target_tensor.pop()
        if fill_na is True:
            target_tensor[-1] = target_tensor[-1].nan_to_num(self.fill_value)
            
        target_tensor = torch.stack(target_tensor, dim = -1).squeeze()
            
            # target = [torch.from_numpy(target_t0_lat).to(torch.float32),
            #     torch.from_numpy(target_t0_lon).to(torch.float32),
            #     torch.from_numpy(target_t0_dtm).to(torch.float32),
            #     torch.from_numpy(target_t0_values).to(torch.float32).nan_to_num(self.fill_value)
            #     ]
            
        
        
        # target_mask = torch.from_numpy(target_t0_mask).to(torch.bool)
        
        return target_tensor, target_mask
    
    def get_weather_video(self, start_date, end_date):
        weather_video = self.weather_xr.sel(time = slice(start_date,
                                                    end_date)) #slice include extremes
        
        W_video = weather_video.to_array().values
        W_video = torch.from_numpy(W_video).to(torch.float32)
        
        weather_coords = torch.from_numpy(self.weather_coords).to(torch.float32)
        weather_coords = torch.moveaxis(weather_coords, -1, 0).unsqueeze(1).expand(-1,W_video.shape[1],-1,-1)
        W_video = torch.cat([weather_coords,
                                    W_video], axis=0)
    
        weather_doy_sin = np.sin((2 * np.pi * weather_video.time.dt.dayofyear.values)/366) 
        weather_doy_cos = np.cos((2 * np.pi * weather_video.time.dt.dayofyear.values)/366) 
        #weather_years = weather_video.time.dt.year.values
        
        W_doy_sin = torch.from_numpy(weather_doy_sin).to(torch.float32)
        W_doy_cos = torch.from_numpy(weather_doy_cos).to(torch.float32)
        #W_years = torch.from_numpy(weather_years).to(torch.float32)
        W_date = torch.stack([W_doy_sin, W_doy_cos], dim = -1)
        
        
        
        return [W_video, W_date]

     
    
if __name__ == "__main__":
    config = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        config = json.load(f)
    print(f"Read data.json: {config}")

    ds = Dataset_2D(config)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")



