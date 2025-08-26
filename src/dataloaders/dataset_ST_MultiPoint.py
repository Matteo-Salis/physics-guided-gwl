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

import torch
from torch.utils.data import Dataset
import torch.nn as nn

import warnings

class Dataset_ST_MultiPoint(Dataset):
    """Weather and Groundwater Dataset for the continuous case model"""

    ### INIT 

    def __init__(self, config):
        """
        Args:
            config (string): Path to the .nc file.
            transform (callable, optional): Optional transform to be applied
                    on a sample.
        """
        
        # Attributes init
        self.config = config

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
        self.loading_point_wtd()
        print("Done!")
        
        if self.config["normalization"] is True:
            
            self.normalize(date_max = np.datetime64(self.config["date_max_norm"]))
            
        print("    Building lagged dataframe...", end = " ")
        self.build_lagged_df()
        print("Done!")
        
    ### Data loading
        
    def loading_weather(self):
        self.weather_xr = xarray.open_dataset(self.config["weather_nc_path"])
        
        self.weather_xr = self.weather_xr.resample(time=self.config["frequency"], label = "left").mean()
        
        self.weather_xr = self.weather_xr.rio.write_crs("epsg:4326")
        self.weather_xr = self.weather_xr[self.config["weather_variables"]]
        
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
            
    def loading_dtm(self):
        self.dtm_roi = rioxarray.open_rasterio(self.config["dtm_nc"],
                                               engine='fiona')
        self.dtm_roi = self.dtm_roi.rio.write_crs("epsg:4326")
    
    def loading_point_wtd(self):
        
        # Water Table Depth data loading
        self.wtd_df = pd.read_csv(self.config["wtd_csv_path"], 
                                    dtype= {"sensor_id": "str"})
        self.wtd_df = self.wtd_df.astype({"date":'datetime64[ns]'})

        # Water Table Depth Sensors shapefile loading: 
        self.wtd_geodf = gpd.read_file(self.config["wtd_shp"],
                                             engine='fiona')
        self.wtd_geodf = self.wtd_geodf.to_crs('epsg:4326')
        
        # Subset Stations
        if self.config["discard_sensor_list"] is not None:

            self.wtd_geodf = self.wtd_geodf.loc[~self.wtd_geodf["sensor_id"].isin(self.config["discard_sensor_list"]), :]
            self.wtd_df = self.wtd_df.loc[~self.wtd_df["sensor_id"].isin(self.config["discard_sensor_list"]), :]
        
        
        if self.config["sel_sensor_list"] is not None:
            self.wtd_geodf = self.wtd_geodf.loc[self.wtd_geodf["sensor_id"].isin(self.config["sel_sensor_list"]), :]
            self.wtd_df = self.wtd_df.loc[self.wtd_df["sensor_id"].isin(self.config["sel_sensor_list"]), :]
        
        # Resampling
        if self.config["frequency"] != "D":
            self.wtd_df.sort_values(by='date', inplace = True)
            self.wtd_df = self.wtd_df.set_index(["date"])
            self.wtd_df = self.wtd_df.groupby([pd.Grouper(freq=self.config["frequency"], label = "left"), "sensor_id"]).mean()
            self.wtd_df = self.wtd_df.reset_index()
            
        # Define attributes about dates and coordinates
        #self.all_dates = self.wtd_df["date"].unique()
        self.sensor_id_list = self.wtd_df["sensor_id"].unique()
        
        # use same sensors order
        self.wtd_geodf["sensor_id"] = pd.Categorical(self.wtd_geodf["sensor_id"], ordered=True, categories=self.sensor_id_list)
        self.wtd_geodf = self.wtd_geodf.sort_values('sensor_id')
        self.wtd_geodf = self.wtd_geodf.reset_index(drop=True)
        
        # Find sensors' heights 
        dtm_values = [self.dtm_roi.sel(x = self.wtd_geodf.geometry.x.values[i],
                             y = self.wtd_geodf.geometry.y.values[i],
                             method = "nearest").values.squeeze() for i in range(len(self.sensor_id_list))]
                                
        self.wtd_geodf["height"] = np.array(dtm_values).squeeze() #add dtm values in the geopandas
        
        ## Build lagged_ds with target and features
        # attach height
        self.wtd_geodf['lon'] = self.wtd_geodf.geometry.x
        self.wtd_geodf['lat'] = self.wtd_geodf.geometry.y

        # Merge coordinates into df using sensor_id
        self.wtd_df = self.wtd_df.merge(self.wtd_geodf[['sensor_id', 'lat', 'lon', 'height']], on='sensor_id', how='left')
        self.wtd_df = self.wtd_df.set_index(['date', 'sensor_id'])
        
        # Define Target
        if self.config["piezo_head"] is True:
            self.compute_piezo_head()
            self.target = "h"
        else:
            self.target = "wtd"
            
        if self.config["relative_target"] is True:
            self.relative_target()
            
        self.target_coords = self.wtd_geodf[["lat","lon","height"]].values
    
    
    ### Normalization
        
    def Euclidean(self,x1,x2,y1,y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    
    def IDW(self, data, LAT, LON, var, beta=2):
        array = np.empty((LAT.shape[0], LON.shape[0]))

        for i, lat_i in enumerate(LAT):
            for j, lon_j in enumerate(LON):
                weights = data.apply(lambda row: self.Euclidean(row.geometry.x, lon_j, row.geometry.y, lat_i)**(-beta), axis = 1)
                z = sum(weights*data[var].values)/weights.sum()
                array[i,j] = z
        return array
    
    def NN(self, data, LAT, LON, var):
        array = np.empty((LAT.shape[0], LON.shape[0]))

        for i, lat_i in enumerate(LAT):
            for j, lon_j in enumerate(LON):
                idx = data.apply(lambda row: self.Euclidean(row.geometry.x, lon_j, row.geometry.y, lat_i), axis = 1).argmin() 
                array[i,j] = data.loc[idx, var]
        return array
            
    def compute_norm_factors(self, date_max = np.datetime64("2020-01-01"), verbose = True, dict_out = False):
        
        subset_wtd_df = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= date_max,
                                                        :]] #
        #subset_wtd_df = subset_wtd_df.loc[subset_wtd_df["nan_mask"] == True, :] #compute only for not nan values
        subset_weather_xr = self.weather_xr.sel(time = slice(date_max)) #slice include extremes
            
        #Subset wrt Date
        if self.config["target_norm_type"] == "sensor_zscore":
            
            # Compute rasterized NN norm factors 
            target_means = subset_wtd_df[self.target].groupby(level=1).transform('mean').values
            target_means = target_means.reshape(len(subset_wtd_df.index)//len(self.sensor_id_list),
                                                len(self.sensor_id_list))[0,:] 
            target_stds = subset_wtd_df[self.target].groupby(level=1).transform('std').values
            target_stds = target_stds.reshape(len(subset_wtd_df.index)//len(self.sensor_id_list),
                                                len(self.sensor_id_list))[0,:]
            
            target_means_gpd = gpd.GeoDataFrame({"mean": target_means}, geometry=self.wtd_geodf.geometry).set_crs(self.wtd_geodf.crs)
            target_stds_gpd = gpd.GeoDataFrame({"std": target_stds}, geometry=self.wtd_geodf.geometry).set_crs(self.wtd_geodf.crs)
            
            bbox = [self.dtm_roi.x.min().values,
                    self.dtm_roi.x.max().values,
                    self.dtm_roi.y.min().values,
                    self.dtm_roi.y.max().values]

            LON = np.linspace(bbox[0], bbox[1], self.config["upsampling_dim"][2])
            LAT = np.linspace(bbox[2], bbox[3], self.config["upsampling_dim"][1])[::-1]
            
            target_means_raster = self.NN(target_means_gpd, LAT, LON, "mean")
            target_stds_raster = self.NN(target_stds_gpd, LAT, LON, "std")
            
            self.target_means_xr = xarray.DataArray(data = target_means_raster,
                                    coords = dict(
                                                lat=("lat", LAT),
                                                lon=("lon", LON),
                                                ),
                                    dims = ["lat", "lon"]
                                    )
            
            self.target_stds_xr = xarray.DataArray(data = target_stds_raster,
                                    coords = dict(
                                                lat=("lat", LAT),
                                                lon=("lon", LON),
                                                ),
                                    dims = ["lat", "lon"]
                                    )
        elif self.config["target_norm_type"] == "overall_zscore":
            
            target_means = subset_wtd_df[self.target].mean()
            target_stds = subset_wtd_df[self.target].std()
            
        elif self.config["target_norm_type"] is None:
            target_means = None
            target_stds = None
        # Use true sensor's stats!!!
        # target_means = []
        # target_stds = []
        # for sensor in self.sensor_id_list:
        #     geom = self.wtd_geodf.loc[self.wtd_geodf["sensor_id"] == sensor].geometry
        #     target_means.append(self.target_means_xr.sel(lon = geom.x.values, lat = geom.y.values, method="nearest").values.flatten())
        #     target_stds.append(self.target_stds_xr.sel(lon = geom.x.values, lat = geom.y.values, method="nearest").values.flatten())
            
        # target_means = np.array(target_means).flatten()
        # target_stds = np.array(target_stds).flatten()
        
        dtm_mean = self.dtm_roi.mean()
        dtm_std = self.dtm_roi.std()
        lat_mean = self.weather_coords.mean(axis=(0,1))[0]
        lat_std = self.weather_coords.std(axis=(0,1))[0]
        lon_mean = self.weather_coords.mean(axis=(0,1))[1]
        lon_std = self.weather_coords.std(axis=(0,1))[1]
        weather_mean = subset_weather_xr.mean()
        weather_std = subset_weather_xr.std()
        
        self.norm_factors = {"target_means": target_means,
                            "target_stds": target_stds,
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
        #self.wtd_df[self.target] = (self.wtd_df[self.target] - self.norm_factors["target_mean"])/self.norm_factors["target_std"]
        
        
        if self.config["target_norm_type"] is not None:
            
            if self.config["target_norm_type"] == "sensor_zscore":
                target_norm_means = np.tile(self.norm_factors["target_means"], len(self.wtd_df.index)//len(self.sensor_id_list))
                target_norm_stds = np.tile(self.norm_factors["target_stds"], len(self.wtd_df.index)//len(self.sensor_id_list))
            
            elif self.config["target_norm_type"] == "overall_zscore":
                target_norm_means = self.norm_factors["target_means"]
                target_norm_stds = self.norm_factors["target_stds"]
            
                
            self.wtd_df[self.target] = (self.wtd_df[self.target] - target_norm_means) / target_norm_stds
        
            
        self.wtd_df["lat"] = (self.wtd_df["lat"] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.wtd_df["lon"] = (self.wtd_df["lon"] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        #self.wtd_data_raserized[self.target] = (self.wtd_data_raserized[self.target] - self.norm_factors["target_mean"])/self.norm_factors["target_std"]
            
        self.dtm_roi = (self.dtm_roi - self.norm_factors["dtm_mean"])/self.norm_factors["dtm_std"]
        self.wtd_df["height"] = (self.wtd_df["height"] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        
        self.target_coords[:,0] = (self.target_coords[:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.target_coords[:,1] = (self.target_coords[:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        self.target_coords[:,2] = (self.target_coords[:,2] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        
        norm_weather_var = list(self.weather_xr.keys())
        norm_weather_var = [var for var in norm_weather_var if var not in ["tmax","tmin"]]
        self.weather_xr[norm_weather_var] = (self.weather_xr[norm_weather_var] - self.norm_factors["weather_mean"][norm_weather_var])/self.norm_factors["weather_std"][norm_weather_var]
        
        if "tmin" in self.config["weather_variables"]:
            self.weather_xr["tmin"] = (self.weather_xr["tmin"] - self.norm_factors["weather_mean"]["tmean"])/self.norm_factors["weather_std"]["tmean"]
        if "tmax" in self.config["weather_variables"]:
            self.weather_xr["tmax"] = (self.weather_xr["tmax"] - self.norm_factors["weather_mean"]["tmean"])/self.norm_factors["weather_std"]["tmean"]
        
        self.weather_coords[:,:,0] = (self.weather_coords[:,:,0] - self.norm_factors["lat_mean"])/self.norm_factors["lat_std"]
        self.weather_coords[:,:,1] = (self.weather_coords[:,:,1] - self.norm_factors["lon_mean"])/self.norm_factors["lon_std"]
        self.weather_coords[:,:,2] = (self.weather_coords[:,:,2] - self.norm_factors["dtm_mean"].values)/self.norm_factors["dtm_std"].values
        
    ### Lagged Dataframe and Temporal Encoding
    
    def temporal_encoding(self, mode, dates):
        
        if mode == "OHE":
            pass
        if mode == "int":
            pass
        if mode == "sin":
            doy_sin = np.sin((2 * np.pi * dates.dayofyear.values)/366) 
            doy_cos = np.cos((2 * np.pi * dates.dayofyear.values)/366)
            temp_enc_names = ["doy_sin","doy_cos"]
            temp_enc = [doy_sin, doy_cos]
            #year_sin = np.cos((2 * np.pi * self.wtd_df.index.get_level_values(0).year.values - 2000)/10)
        
        return temp_enc, temp_enc_names
    
    def build_lagged_df(self):
        
        (doy_sin, doy_cos), self.temp_enc_names = self.temporal_encoding(mode = self.config["temporal_encoding"], dates = self.wtd_df.index.get_level_values(0))
        
        self.wtd_df[self.temp_enc_names[0]] = doy_sin
        self.wtd_df[self.temp_enc_names[1]] = doy_cos
        
        # Create lagged features
        self.target_lags = self.config["target_lags"]
        
        # Unstack to get features: each column is a sensor_id, each row is a date
        features = self.wtd_df[self.target].unstack(level=1)
        features_temp_enc = self.wtd_df.loc[(features.index,self.sensor_id_list[0]), self.temp_enc_names].droplevel(1)
        features = pd.concat([features_temp_enc, features], axis=1)
        
        lagged_features = []
        lag_names = []
        avail_mask_names = []
        for lag in self.target_lags:
            
            temp_lagged_ds = features.shift(lag).add_suffix(f'_lag{lag}')
            lag_names.extend(list(temp_lagged_ds.columns))
            avail_mask_names.append(f"avail_mask_lag{lag}")
            temp_lagged_ds[avail_mask_names[-1]] = (len(self.sensor_id_list) - temp_lagged_ds.isna().sum(axis=1)) > self.config["nan_treshold"]
            lagged_features.append(temp_lagged_ds)
        
        features_lagged = pd.concat(lagged_features, axis=1)
        features_lagged["avail_mask"] = features_lagged[avail_mask_names].all(axis = 1)
        
        # Reconstruct target by stacking the original dataframe
        target_data = self.wtd_df[[self.target[0],"lat","lon","height",*self.temp_enc_names]]

        # Build final dataset: join features with target (reset index to align properly)
        features_lagged_reindex = features_lagged.loc[target_data.index.get_level_values(0)].reset_index(drop=True)
        target_data = target_data.reset_index(drop=False)

        self.lagged_df = pd.concat([target_data, features_lagged_reindex], axis=1)
        
        # Delete instance with no lagged measurement
        self.lagged_df = self.lagged_df.loc[self.lagged_df["avail_mask"]]
        avail_mask_names.append("avail_mask")
        self.lagged_df = self.lagged_df.drop(avail_mask_names,
                                       axis = 1)
        
        self.lag_names = lag_names
        self.features_names = ["lat","lon","height",
                               *self.temp_enc_names,
                               *self.lag_names]
        
        self.lagged_df = self.lagged_df.set_index(['date', 'sensor_id'])
        
        # Create nan-mask
        self.lagged_df["nan_mask"] = self.lagged_df[self.target].isna()
        
        self.dates = self.lagged_df.index.get_level_values(0).unique()
        
        if self.config["fill_value"]:
            self.fill_value = self.config["fill_value"]
        else:
            self.fill_value = 0
    
    
    ### Utilities
    
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

            
    # #def sparse_target_coords_object(self):
    #     self.sparse_target_coords = self.wtd_df.loc[pd.IndexSlice[self.input_dates[0], :],
    #                                                 ["lat","lon","height"]]
        
    #     z_lat = self.sparse_target_coords["lat"].values
    #     z_lon = self.sparse_target_coords["lon"].values
    #     z_height = self.sparse_target_coords["height"].values
        
    #     self.sparse_target_coords = [z_lat,
    #                                 z_lon,
    #                                 z_height]
        
    #     self.sparse_target_coords = np.stack(self.sparse_target_coords, axis = -1).squeeze()
        
    def compute_piezo_head(self):
        self.wtd_df["h"] = self.wtd_df["height"] - self.wtd_df["wtd"]
        
    def relative_target(self):
        self.wtd_df[f"{self.target}_r"] = self.wtd_df[self.target]/self.wtd_df["height"]
        self.target = f"{self.target}_r"
        
    def get_iloc_from_date(self, date_max):
        """
        return iloc of last sensor before date_max
        """
        
        # iloc = self.lagged_df.index.get_loc(self.lagged_df[ds.lagged_df.index.get_level_values(0) <= date_max].iloc[-1].name)
        # return iloc
        return np.argwhere(self.dates <= date_max).max()

    def __len__(self):
        return len(self.dates)
    
    
    ### GET
    
    def get_target_values(self, subset_df):
        
        target = subset_df[self.target].values
        target_nan_mask = subset_df["nan_mask"].values
        
        Y = [torch.from_numpy(target).to(torch.float32),
             torch.from_numpy(target_nan_mask).to(torch.bool)]
        
        return Y
    
    def get_target_st_info(self, subset_df):
        target_st_info = subset_df[["lat","lon","height",*self.temp_enc_names]].values
        
        Z = torch.from_numpy(target_st_info).to(torch.float32)
        
        return Z
    
    
    def get_lagged_features(self, subset_df):
        
        lagged_features = subset_df[self.lag_names].iloc[0] # same data for all rows for that date
        
        target_lags_values = lagged_features[~lagged_features.index.str.contains("|".join(self.temp_enc_names), regex=True)].values
        target_lags_values = target_lags_values.reshape((len(self.target_lags),len(self.sensor_id_list)))
        
        target_lags_nan_mask = np.isnan(target_lags_values)
        
        ## spatio-temporal info
        temp_encoding_lags = lagged_features[lagged_features.index.str.contains("|".join(self.temp_enc_names), regex=True)].values
        temp_encoding_lags = temp_encoding_lags.reshape((len(self.target_lags),len(self.temp_enc_names)))
        
        target_lags_st_info = []
        for i in range(len(self.target_lags)):
            temp_encoding_expand = np.repeat(temp_encoding_lags[i,:][None,:],
                                             len(self.sensor_id_list), axis=0)
            
            target_lags_st_info.append(np.concat([self.target_coords,
                                                  temp_encoding_expand], axis = 1))
            
        target_lags_st_info = np.stack(target_lags_st_info, axis = 0) # (D,S,C)
        
        X = [torch.from_numpy(target_lags_values).to(torch.float32).nan_to_num(self.fill_value),
             torch.from_numpy(target_lags_st_info).to(torch.float32),
             torch.from_numpy(target_lags_nan_mask).to(torch.bool)]
        
        return X
        
        
    def get_weather_features(self, target_date):
        
        start_weather_date = target_date - np.timedelta64(self.config["weather_lags"], self.config["frequency"])
        weather_video = self.weather_xr.sel(time = slice(start_weather_date,
                                                         target_date))
        weather_video_dates = weather_video.time.dt
        weather_video = weather_video.to_array().values
        
        ## spatio-temporal info
        weather_coords = np.moveaxis(self.weather_coords, -1, 0)[:,None,:,:]
        weather_coords = np.tile(weather_coords, (1,weather_video.shape[1],1,1))
        
        (weather_doy_sin, weather_doy_cos), _ = self.temporal_encoding(mode = self.config["temporal_encoding"], dates = weather_video_dates)
        weather_t_info = np.stack([weather_doy_sin, weather_doy_cos], axis = 1)
        
        weather_t_info = np.tile(weather_t_info[:,None,None,:], (1,
                                                self.weather_coords.shape[0],
                                                self.weather_coords.shape[1],
                                                1))
        weather_t_info = np.moveaxis(weather_t_info, -1, 0)
        weather_st_info = np.concat([weather_coords,weather_t_info],
                                    axis = 0)
        
        W = [torch.from_numpy(weather_video).to(torch.float32),
             torch.from_numpy(weather_st_info).to(torch.float32),]
        
        return W
        
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
            
        target_date = np.datetime64(self.dates[idx]) #.astype(f"datetime64[{self.config['frequency']}]")
            
        subset_df = self.lagged_df.loc[pd.IndexSlice[target_date,:],:]
            
        # Get target Y, Z
        Y = self.get_target_values(subset_df)
        Z = self.get_target_st_info(subset_df)
        
        # Get features X
        ## lagged 
        X = self.get_lagged_features(subset_df)
        
        # Weather W
        W = self.get_weather_features(target_date)
        
        return [X,
                W,
                Z,
                Y]
     
    
if __name__ == "__main__":
    config = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        config = json.load(f)
    print(f"Read data.json: {config}")

    ds = Dataset_ST_MultiPoint(config)
    print("Dataset created.")
    print(f"Length of the dataset: {ds.__len__()}")
    print(f"Item -1: {ds[-1]}")



