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


class Dataset_2D(Dataset):
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
        self.loading_weather()
        
        # Digital Terrain Model data loading
        self.loading_dtm()
        
        # Water Table Depth data loading 
        self.loading_point_wtd(fill_value = config["fill_value"])
        
        if config["piezo_head"] is True:
            self.compute_piezo_head()
            self.target = "h"
        else:
            self.target = "wtd"
            
        # Rasterizing groundwater dataset 
        self.rasterize_sparse_measurements(downscale_factor = config["target_downscale_factor"])
        # 0.0437 around 2km resolution
        
        if config["normalization"] is True:
            
            self.normalize(date_max = np.datetime64(config["date_max_norm"]))
            
        # Usefull in training 
        self.weather_coords_dtm = self.get_weather_coords(dtm = True)

        # Transform
        # if config["densification_dropout"]:
        #     self.densification_dropout = partial(self.densification_dropout, perc_dropout = config["densification_dropout"])
        
        
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
        
        self.wtd_data_raserized = (self.wtd_data_raserized - self.norm_factors["target_mean"])/self.norm_factors["target_std"]
            
        self.dtm_roi = (self.dtm_roi - self.norm_factors["dtm_mean"])/self.norm_factors["dtm_std"]
        self.wtd_df["height"] = (self.wtd_df["height"] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        self.weather_dtm = (self.weather_dtm - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
            
        self.weather_coords[:,:,0] = (self.weather_coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.weather_coords[:,:,1] = (self.weather_coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        
        self.dtm_coords[:,:,0] = (self.dtm_coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.dtm_coords[:,:,1] = (self.dtm_coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        self.dtm_coords[:,:,2] = (self.dtm_coords[:,:,2] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
            
        self.weather_xr = (self.weather_xr - self.norm_factors["weather_mean"])/self.norm_factors["weather_std"]       
            
        
    def loading_dtm(self):
        self.dtm_roi = rioxarray.open_rasterio(self.config["dtm_nc"],
                                               engine='fiona')
        self.dtm_roi = self.dtm_roi.rio.write_crs("epsg:4326")
        
        self.dtm_coords = self.coordinates_xr(self.dtm_roi, coord_name = "xy")
        
        self.dtm_coords = np.concat([self.dtm_coords,
                                     np.moveaxis(self.dtm_roi.values, 0,-1)],
                                    axis=-1)
        
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
        self.weather_xr = self.weather_xr.rio.write_crs("epsg:4326")
        
        self.weather_coords = self.coordinates_xr(self.weather_xr)
        
        self.weather_dtm = rioxarray.open_rasterio(self.config["weather_dtm"],
                                               engine='fiona')
        
        self.weather_dtm = self.weather_dtm.values
        self.weather_dtm = np.moveaxis(self.weather_dtm, 0,-1)
        
    def rasterize_sparse_measurements(self, downscale_factor):
        # downscaling dtm
        
        new_width = round(self.dtm_roi.rio.width * downscale_factor)
        new_height = round(self.dtm_roi.rio.height * downscale_factor)

        self.dtm_roi_downsampled = self.dtm_roi.rio.reproject(
            self.dtm_roi.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )

        wtd_df_util = copy.deepcopy(self.wtd_df)
        wtd_df_util["date"] = self.wtd_df.index.get_level_values(0)
        wtd_data_geop = gpd.GeoDataFrame(
        wtd_df_util, geometry =gpd.points_from_xy(wtd_df_util["lon"], 
                                            wtd_df_util["lat"]), crs="EPSG:4326")
        wtd_data_geop = wtd_data_geop[["date",self.target,"geometry"]]

        self.minx = self.dtm_roi_downsampled.x.min()
        self.miny = self.dtm_roi_downsampled.y.min()
        self.maxx = self.dtm_roi_downsampled.x.max()
        self.maxy = self.dtm_roi_downsampled.y.max()
        
        print("Rasterizing groundwater dataframe...")
        rasterized_ds_list = []
        for date_idx in range(len(self.dates)):
            
            vector_ds = wtd_data_geop.loc[wtd_data_geop["date"] == self.dates[date_idx],:]

            rasterized_ds = make_geocube(vector_data=vector_ds,
                                        measurements=[self.target],
                                        output_crs="epsg:4326",
                                        # resolution= new_resolution,
                                        resolution=(self.dtm_roi_downsampled.rio.transform().a, round(self.dtm_roi_downsampled.rio.transform().e, 4)),
                                        # Global extent in degrees of longitude and latitude
                                        geom=box(minx=self.minx, miny=self.miny, maxx=self.maxx, maxy=self.maxy))
            
            rasterized_ds_list.append(rasterized_ds)
        print("Rasterized groundwater dataframe.")

        self.wtd_data_raserized = xarray.concat(rasterized_ds_list, dim = "time")
        self.wtd_data_raserized = self.wtd_data_raserized.assign_coords({"time": self.dates})

        self.wtd_data_raserized = self.wtd_data_raserized.reindex(y=list(reversed(self.wtd_data_raserized.y)))
        self.wtd_data_raserized = self.wtd_data_raserized.reindex(x=list(reversed(self.wtd_data_raserized.x)))

        mask = self.wtd_data_raserized.notnull()
        mask = mask.rename({self.target:'nan_mask'})
        self.wtd_data_raserized = self.wtd_data_raserized.assign(mask = mask["nan_mask"])
        self.target_rasterized_numpy = self.wtd_data_raserized.to_array().values.astype(np.float32)
        
        self.target_rasterized_dtm = self.dtm_roi.sel(x = self.wtd_data_raserized.x,
                                                         y = self.wtd_data_raserized.y,
                                                         method = "nearest")
        self.target_rasterized_coords = self.coordinates_xr(self.target_rasterized_dtm, coord_name = "xy")
        
        self.target_rasterized_coords = np.concat([self.target_rasterized_coords,
                                    np.moveaxis(self.target_rasterized_dtm.values, 0,-1)],
                                    axis=-1)

    def loading_point_wtd(self, fill_value = None):
        
        # Water Table Depth data loading
        self.wtd_df = pd.read_csv(self.config["wtd_csv_path"], 
                                    dtype= {"sensor_id": "str"})
        self.wtd_df = self.wtd_df.astype({"date":'datetime64[ns]'})

        # Water Table Depth Sensors shapefile loading: 
        self.wtd_names = gpd.read_file(self.config["wtd_shp"],
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
        
        # Subset wtd data truncating the last `twindow` instances
        last_date = self.dates.max() - np.timedelta64(self.twindow, 'D')
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
        
        max_idx = np.argmax(self.wtd_data_raserized.time.values[self.wtd_data_raserized.time.values < date_max])
        #row_num = self.wtd_df.index.get_loc(self.wtd_df[self.wtd_df.index.get_level_values(0) < date_max].iloc[-1].name)
        return max_idx
        
    def __len__(self):
        # data = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= self.input_dates.max(),
        #                                                :]]
        # return len(data)
        
        return len(self.wtd_data_raserized[self.target]) - self.twindow
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
        
        # Retrieve date and coords for idx instance
        start_date = self.wtd_data_raserized.time.values[idx] #self.wtd_df.iloc[idx, :].name[0]
        end_date = start_date + np.timedelta64(self.twindow, "D")
        
        # idx_icon = np.argmin(self.wtd_df.index[self.wtd_df.index.get_level_values(0) == start_date])
        
        # sample_lat = self.wtd_df["lat"].iloc[idx:idx_icon+len(self.sensor_id_list)]
        # sample_lon = self.wtd_df["lon"].iloc[idx_icon:idx_icon+len(self.sensor_id_list)]
        # sample_dtm = self.wtd_df["height"].iloc[idx_icon:idx_icon+len(self.sensor_id_list)]
        
        # Z = [torch.tensor(sample_lat).reshape(1).to(torch.float32),
        #      torch.tensor(sample_lon).reshape(1).to(torch.float32),
        #      torch.tensor(sample_dtm).reshape(1).to(torch.float32)]
          
        # Z = torch.stack(Z, dim = -1).squeeze()
        Z = torch.from_numpy(self.target_rasterized_coords).to(torch.float32)
        
        # Initial state WTD (t0) data        
        X, X_mask = self.get_icon_target_data(start_date)
        
        # Retrieve weather data
        W = self.get_weather_video(start_date, end_date)
        
        # Retrieve wtd values from t0+1 to T for the idx instance sensor
        #Y, Y_mask = self.get_target_series(start_date, end_date, sample_lat, sample_lon)
        Y = torch.from_numpy(self.target_rasterized_numpy[0,idx+1:idx+1+self.twindow,:,:]).to(torch.float32)
        Y_mask = torch.from_numpy(self.target_rasterized_numpy[1,idx+1:idx+1+self.twindow,:,:]).to(torch.bool)
        
        # if self.densification_dropout:
        #     X, X_mask = self.densification_dropout([X, X_mask])
        
        return [X, Z, W, Y, X_mask, Y_mask]
    
    # def densification_dropout(self, sample, perc_dropout = 0.25):
        
    #     """
    #     Dropout training as in densification of Andrychowicz et al. (2023)
    #     """
        
    #     new_X, new_X_mask = copy.deepcopy(sample)
    #     dropout_mask = torch.rand(new_X_mask.shape, device=new_X.device) > perc_dropout
        
    #     new_X[:,-1] = new_X[:,-1] * dropout_mask
    #     new_X_mask = torch.logical_and(new_X_mask, dropout_mask)
        
    #     return new_X, new_X_mask
    
    def get_icon_target_data(self, date):
        
        target_t0 = self.wtd_df[[self.target, "nan_mask", "lat", "lon", "height"]].loc[self.wtd_df.index.get_level_values(0) == date]
        
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
        X_mask = torch.from_numpy(target_t0_mask).to(torch.bool)
        
        return X, X_mask
    
    def get_target_series(self, start_date, end_date, lat, lon):
        target_t1_T = self.wtd_df[[self.target, "nan_mask"]].loc[(self.wtd_df.index.get_level_values(0) > start_date) &
                                          (self.wtd_df.index.get_level_values(0) <= end_date)  & 
                                          (self.wtd_df["lat"] == lat) &
                                          (self.wtd_df["lon"] == lon)]
        
        target_t1_T_values =  target_t1_T[self.target].values
        target_t1_T_mask =  target_t1_T["nan_mask"].values        
        
        Y = torch.from_numpy(target_t1_T_values).to(torch.float32)
        Y_mask = torch.from_numpy(target_t1_T_mask).to(torch.bool)
        
        return Y, Y_mask
    
    def get_weather_video(self, start_date, end_date):
        weather_video = self.weather_xr.sel(time = slice(start_date + np.timedelta64(1, "D"),
                                                    end_date)) #slice include extremes
        
        weather_doy = np.sin((2 * np.pi * weather_video.time.dt.dayofyear.values)/366) 
        weather_years = weather_video.time.dt.year.values
        
        weather_video = weather_video.to_array().values
        
        W_video = torch.from_numpy(weather_video).to(torch.float32)
        
        W_doy = torch.from_numpy(weather_doy).to(torch.float32)
        W_years = torch.from_numpy(weather_years).to(torch.float32)
        W_date = torch.stack([W_doy, W_years], dim = -1)
        
        return [W_video, W_date]
    
    def get_weather_dtm(self):
        return torch.from_numpy(self.weather_dtm).to(torch.float32)
        
    def get_weather_coords(self, dtm = False):
        
        output = torch.from_numpy(self.weather_coords).to(torch.float32)
        
        if dtm is True:
            dtm = self.get_weather_dtm()
            output = torch.cat([output, dtm], dim = -1)

        return output
    
    def control_points_generator(self,
                                 bbox = None,
                                 mode = "even",
                                 num_lon_point = 100,
                                 num_lat_point = 100,
                                 step = None, 
                                 normalized = True,
                                 flatten = True):
        """
        output: normalized cpoints
        """

        if bbox is None:
            bbox = [self.dtm_roi.x.min().values,
                    self.dtm_roi.x.max().values,
                    self.dtm_roi.y.min().values,
                    self.dtm_roi.y.max().values]
            
        if mode == "even":
            
            # create one-dimensional arrays for x and y
            x = np.linspace(bbox[0], bbox[1], num_lon_point)
            y = np.linspace(bbox[2], bbox[3], num_lat_point)[::-1]
            # create the mesh based on these arrays
            X, Y = np.meshgrid(x, y)
            coords = np.stack([Y, X], axis = -1)
            
            
            dtm_xy = self.dtm_roi.sel(x = x, y = y,
                            method = "nearest").values
            
            if normalized is True:
                coords[:,:,0] = (coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                coords[:,:,1] = (coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
            else:
                dtm_xy = (dtm_xy * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
            
            coords = np.concat([coords, np.moveaxis(dtm_xy, 0, -1)], axis=-1)
            
            if flatten is True:
                coords = coords.reshape(coords.shape[0]*coords.shape[1], coords.shape[2])
            
            return coords
            
        elif mode == "urandom":
            
            if(num_lon_point != num_lat_point):
                warnings.warn("number of lat cpoints not equal to lon cpoints... the min is considered in the following")
            num_cpoints = min(num_lon_point, num_lat_point)
            
            x = np.random.uniform(low=bbox[0], high=bbox[1], size=num_cpoints)
            y = np.random.uniform(low=bbox[2], high=bbox[3], size=num_cpoints)
            
            dtm_xy = np.array([self.dtm_roi.sel(x = x[i], y = y[i],
                            method = "nearest").values for i in range(num_cpoints)])
            
            if normalized is True:
                y = (y - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                x = (x - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
            else:
                dtm_xy = (dtm_xy * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
            
            coords = np.concat([np.expand_dims(y, 1), np.expand_dims(x, 1), dtm_xy], axis=-1)
            
            return coords
            
        elif mode == "urandom+nb":
            
            if(num_lon_point != num_lat_point):
                warnings.warn("number of lat cpoints not equal to lon cpoints... the min is considered in the following")
            num_cpoints = min(num_lon_point, num_lat_point)
            
            margin = 1.2 * step
            x = np.random.uniform(low=bbox[0]+margin, high=bbox[1]-margin, size=num_cpoints)
            y = np.random.uniform(low=bbox[2]+margin, high=bbox[3]-margin, size=num_cpoints)
            
            x_right = x + step
            x_two_right = x + 2*step
            
            x_left = x - step
            x_two_left = x - 2*step
            
            y_up = y + step
            y_two_up = y + 2*step
            
            y_down = y - step
            y_two_down = y - 2*step
            
            dtm_xy = []
            dtm_xy_right = []
            dtm_xy_two_right = []
            dtm_xy_left = []
            dtm_xy_two_left = []
            dtm_xy_up = []
            dtm_xy_two_up = []
            dtm_xy_down = []
            dtm_xy_two_down = []
            
            for i in range(num_cpoints):
                dtm_xy.append(self.dtm_roi.sel(x = x[i], y = y[i],
                                method = "nearest").values)
                
                dtm_xy_right.append(self.dtm_roi.sel(x = x_right[i], y = y[i],
                                method = "nearest").values)
                dtm_xy_two_right.append(self.dtm_roi.sel(x = x_two_right[i], y = y[i],
                                method = "nearest").values)
                
                dtm_xy_left.append(self.dtm_roi.sel(x = x_left[i], y = y[i],
                                method = "nearest").values)
                dtm_xy_two_left.append(self.dtm_roi.sel(x = x_two_left[i], y = y[i],
                                method = "nearest").values)
                
                dtm_xy_up.append(self.dtm_roi.sel(x = x[i], y = y_up[i],
                                method = "nearest").values)
                dtm_xy_two_up.append(self.dtm_roi.sel(x = x[i], y = y_two_up[i],
                                method = "nearest").values)
                
                dtm_xy_down.append(self.dtm_roi.sel(x = x[i], y = y_down[i],
                                method = "nearest").values)
                dtm_xy_two_down.append(self.dtm_roi.sel(x = x[i], y = y_two_down[i],
                                method = "nearest").values)
            
            # Normalization
            if normalized is True:
                x = (x - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
                x_right = (x_right - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
                x_two_right = (x_two_right - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
                x_left = (x_left - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
                x_two_left = (x_two_left - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
                
                y = (y - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                y_up = (y_up - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                y_two_up = (y_two_up - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                y_down = (y_down - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
                y_two_down = (y_two_down - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
            
            else:
                dtm_xy = (dtm_xy * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                
                dtm_xy_right = (dtm_xy_right * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                dtm_xy_two_right = (dtm_xy_two_right * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                
                dtm_xy_left = (dtm_xy_left * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                dtm_xy_two_left = (dtm_xy_two_left * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                
                dtm_xy_up = (dtm_xy_up * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                dtm_xy_two_up = (dtm_xy_two_up * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                
                dtm_xy_down = (dtm_xy_down * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
                dtm_xy_two_down = (dtm_xy_two_down * self.norm_factors["dtm_std"].values) + self.norm_factors["dtm_mean"].values
    
            dtm_xy = np.array(dtm_xy)
            
            dtm_xy_right = np.array(dtm_xy_right)
            dtm_xy_two_right = np.array(dtm_xy_two_right)
            
            dtm_xy_left = np.array(dtm_xy_left)
            dtm_xy_two_left = np.array(dtm_xy_two_left)
            
            dtm_xy_up = np.array(dtm_xy_up)
            dtm_xy_two_up = np.array(dtm_xy_two_up)
            
            dtm_xy_down = np.array(dtm_xy_down)
            dtm_xy_two_down = np.array(dtm_xy_two_down)
            
            coords = np.concat([np.expand_dims(y, 1), np.expand_dims(x, 1), dtm_xy], axis=-1)
            
            coords_right = np.concat([np.expand_dims(y, 1), np.expand_dims(x_right, 1), dtm_xy_right], axis=-1)
            coords_two_right = np.concat([np.expand_dims(y, 1), np.expand_dims(x_two_right, 1), dtm_xy_two_right], axis=-1)
            
            coords_left = np.concat([np.expand_dims(y, 1), np.expand_dims(x_left, 1), dtm_xy_left], axis=-1)
            coords_two_left = np.concat([np.expand_dims(y, 1), np.expand_dims(x_two_left, 1), dtm_xy_two_left], axis=-1)
            
            coords_up = np.concat([np.expand_dims(y_up, 1), np.expand_dims(x, 1), dtm_xy_up], axis=-1)
            coords_two_up = np.concat([np.expand_dims(y_two_up, 1), np.expand_dims(x, 1), dtm_xy_two_up], axis=-1)
            
            coords_down = np.concat([np.expand_dims(y_down, 1), np.expand_dims(x, 1), dtm_xy_down], axis=-1)
            coords_two_down = np.concat([np.expand_dims(y_two_down, 1), np.expand_dims(x, 1), dtm_xy_two_down], axis=-1)
            
            all_coords = np.stack([coords, coords_right, coords_left, coords_up, coords_down,
                coords_two_right, coords_two_left, coords_two_up, coords_two_down], axis=-1)
            
            return all_coords
    
     
    
if __name__ == "__main__":
    config = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        config = json.load(f)
    print(f"Read data.json: {config}")

    ds = Dataset_2D(config)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")



