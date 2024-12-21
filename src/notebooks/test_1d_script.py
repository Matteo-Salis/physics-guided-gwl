# %% [markdown]
# # Libraries

# %%
from operator import itemgetter
from tqdm import tqdm
import time
from datetime import datetime
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray
import fiona

#import matplotlib
import matplotlib.pyplot as plt

#from rasterio.enums import Resampling

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
from torch.autograd import Variable

import wandb

# %% [markdown]
# # Load dictionary

# %%
json_file = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/discrete_2D_wtd/test_1D.json"
dict_files = {}
with open(json_file) as f:
    dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

# %% [markdown]
# # Dataset class

# %%

class ContinuousDataset(Dataset):
    """Weather and WTD Dataset for the continuous case model"""

    def __init__(self, dict_files, #meteo_nc_path, wtd_csv_path, wtd_stations_shp_path,
                 fill_value = 0,
                 transform = None):
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
        self.loading_point_wtd(fill_value = fill_value)

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
        
        self.weather_dtm = rioxarray.open_rasterio(self.dict_files["weather_dtm"],
                                               engine='fiona')
        
        self.weather_dtm = self.weather_dtm.values
        self.weather_dtm = np.moveaxis(self.weather_dtm, 0,-1)

    def loading_point_wtd(self, fill_value = 0):
        
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
        
        # Subset wtd data truncating the last `timestep` instances
        last_date = self.dates.max() - np.timedelta64(self.timesteps, 'D')
        self.input_dates = self.dates[self.dates <= last_date]
        
        # Create nan-mask
        self.wtd_df["nan_mask"] = 1*~self.wtd_df["wtd"].isna()
        self.wtd_df["wtd"] = self.wtd_df["wtd"].fillna(fill_value)
        
    def __len__(self):
        data = self.wtd_df.loc[pd.IndexSlice[self.wtd_df.index.get_level_values(0) <= self.input_dates.max(),
                                                       :,
                                                       :]]
        return len(data)
    
    def __getitem__(self, idx):
        
        if idx < 0:
            idx = self.__len__() + idx
        
        # Retrieve date and coords for idx instance
        start_date = self.wtd_df.iloc[idx, :].dropna().name[0]
        sample_lat = self.wtd_df.iloc[idx, :].dropna().name[1]
        sample_lon = self.wtd_df.iloc[idx, :].dropna().name[2]
        sample_dtm = self.dtm_roi.sel(x = sample_lon,
                                      y = sample_lat,
                                      method = "nearest").values  
        
        end_date = start_date + np.timedelta64(self.timesteps, "D")
        
        # print("start date: ", str(start_date))
        # print("end date: ", str(end_date))
        
        # Initial state WTD (t0) data
        wtd_t0 = self.wtd_df[["wtd", "nan_mask"]].loc[self.wtd_df.index.get_level_values(0) == start_date].dropna()
        wtd_t0_values = wtd_t0["wtd"].values
        #wtd_t0_mask = wtd_t0["nan_mask"].values
        wtd_t0_lat = wtd_t0.index.get_level_values(1).values
        wtd_t0_lon = wtd_t0.index.get_level_values(2).values
        wtd_t0_dtm = np.array([self.dtm_roi.sel(x = wtd_t0_lon[sensor],
                                                y = wtd_t0_lat[sensor],
                                                method = "nearest") for sensor in range(len(wtd_t0_lat))]).squeeze()
        
        #wtd_t0_mask = 1*~np.isnan(wtd_t0_values)
        X = [torch.from_numpy(wtd_t0_lat).to(torch.float32),
             torch.from_numpy(wtd_t0_lon).to(torch.float32),
             torch.from_numpy(wtd_t0_dtm).to(torch.float32),
             torch.from_numpy(wtd_t0_values).to(torch.float32),
             #torch.from_numpy(wtd_t0_mask).to(torch.float32)
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
        wtd_t1_T = self.wtd_df[["wtd", "nan_mask"]].loc[(self.wtd_df.index.get_level_values(0) > start_date) &
                                          (self.wtd_df.index.get_level_values(0) <= end_date)  & 
                                          (self.wtd_df.index.get_level_values(1) == sample_lat)&
                                          (self.wtd_df.index.get_level_values(2) == sample_lon)]
        
        wtd_t1_T_values =  wtd_t1_T["wtd"].values
        wtd_t1_T_mask =  wtd_t1_T["nan_mask"].values        
        
        Y = [torch.from_numpy(wtd_t1_T_values).to(torch.float32),
             torch.from_numpy(wtd_t1_T_mask).to(torch.float32)]
        
        Y = torch.stack(Y, dim = -1)
        
        if self.transform:
            sample = self.transform(sample)
        
        return [X, Z, W, Y]
    
    def get_weather_dtm(self):
        return torch.from_numpy(self.weather_dtm).to(torch.float32)
        
    def get_weather_coords(self):
        return torch.from_numpy(self.weather_coords).to(torch.float32)

# %%
ds = ContinuousDataset(dict_files)

# %%
print(f"Length of the dataset: {ds.__len__()}")

# %%
ds[0][0][:,:3].shape

# %% [markdown]
# # Model 

# %%
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Continuous1DNN(nn.Module):
    def __init__(self,
                 timestep = 180,
                 cb_fc_layer = 5,
                 cb_fc_neurons = 16,
                 conv_filters = 16,
                 lstm_layer = 3,
                 lstm_input_units = 16,
                 lstm_units = 32):
        super().__init__()
        
        self.timestep = timestep
        self.lstm_layer = lstm_layer
        self.lstm_input_units = lstm_input_units
        self.lstm_units = lstm_units
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        
        self.wgamma = nn.Sigmoid()
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(4, self.cb_fc_neurons))
        cb_fc.append(nn.ReLU())
        for l in range(self.cb_fc_layer - 2):
            cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
            cb_fc.append(nn.ReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.ReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        self.weather_wgamma = nn.Sigmoid()
        
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(10, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.ReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.lstm_input_units, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.lstm_input_units))
        conv3d_stack.append(nn.ReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)


    def forward(self, x, z, w):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        
        # #x (31, 5); z (3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3); y (180, 2)
        # [wtd_t0_lat, wtd_t0_lon,
        #  wtd_t0_dtm, wtd_t0_values,
        #  wtd_t0_mask] = x
        
        # [sample_lat, sample_lon, sample_dtm] = z
        
        # Conditioning block
        
        
        wtd_sim = torch.matmul(x[:,:,:3], z.unsqueeze(-1))
        wtd_sim = self.wgamma(wtd_sim)
        print("sim", wtd_sim.shape)
        
        wtd0 = torch.sum(x[:,:,:-1] * wtd_sim, dim = (1,2))/torch.sum(wtd_sim, dim = (1,2))
        print("wtd0", wtd0.shape)
        print("z", z.shape)
        wtd0 = torch.cat([z, wtd0.unsqueeze(-1)], dim = -1)
        print("wtd0", wtd0.shape)
        
        wtd0 = self.cb_fc(wtd0)
        print("wtd0 emb", wtd0.shape)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_sim = w[1] * z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_sim = torch.sum(weather_sim, dim = -1)
        weather_sim = self.weather_wgamma(weather_sim)
        weather_sim = weather_sim[:, None, None, : ,:].expand(-1, w[0].shape[1], w[0].shape[2], -1, -1 )
        
        weather = w[0] * weather_sim
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        wb_td3dconv = wb_td3dconv.squeeze()
        wb_td3dconv = torch.moveaxis(wb_td3dconv, 1, -1)
        
        # Sequential block
        wtd0 = wtd0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        wtd0 = torch.movedim(wtd0, 0, 1)
        
        wtd_series = self.lstm_1(wb_td3dconv,
                                 (wtd0.contiguous(),
                                  wtd0.contiguous())) #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        wtd_series = self.fc(wtd_series[0])
        
        return wtd_series.squeeze()

model = Continuous1DNN().to(device)

# %%
print("Total number of trainable parameters: " ,sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% [markdown]
# # Training

# %%
batch_size = dict_files["batch_size"]
max_epochs = dict_files["epochs"]

test_split_p = dict_files["test_split_p"]
train_split_p = 1 - test_split_p

max_ds_elems = ds.__len__()
if not dict_files["all_dataset"]:
    max_ds_elems = dict_files["max_ds_elems"]

train_idx = int(max_ds_elems*train_split_p)
test_idx = int(max_ds_elems*test_split_p)

print(f"Traing size: {train_idx}, Test size: {test_idx}")

train_idxs, test_idxs = np.arange(train_idx), np.arange(train_idx, train_idx + test_idx)

train_sampler = SequentialSampler(train_idxs)
test_sampler = SequentialSampler(test_idxs)

train_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset=ds,
                                            batch_size=batch_size,
                                            sampler=test_sampler)

# %%
def plot_predictions(x, y, y_hat, save_dir = None):
    fig, ax = plt.subplots()
    fig.suptitle("Loss vs iterations")
    ax.plot(x, y_hat, label = "predicted")
    ax.plot(x, y, label = "true")
    ax.legend()
    if save_dir:
        plt.savefig(f"{save_dir}.png", bbox_inches = 'tight') #dpi = 400, transparent = True
    else:
        plt.tight_layout()
        plt.show()

# %%
def masked_mse(y_hat, y, mask):
    # y_hat = y_hat.to(device)
    # y = y.to(device)
    # mask = mask.to(device)
    return torch.sum(((y_hat-y)*mask)**2.0)  / torch.sum(mask)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# %%
wandb.init(
    entity="gsartor-unito",
    project=dict_files["experiment_name"],
    dir =dict_files["wandb_dir"],
    config=dict_files,
    mode="offline",
)

# Magic
wandb.watch(model, log_freq=100)

# %%
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#save_dir = "/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/runs/continuous_1d"

weather_coords = ds.get_weather_coords()
weather_dtm = ds.get_weather_dtm()
weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)
weather_coords = weather_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

print('mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
#print(torch.cuda.memory_summary(device=None, abbreviated=False))

for i in range(max_epochs):
    
    model.train(True)
    start_time = time.time()
    print(f"############### Training epoch {i} ###############")
    
    with tqdm(train_loader, unit="batch") as tepoch:
            for batch_idx, (x, z, w_values, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {i}")
                
                x = x.to(device)
                z = z.to(device)
                w = [w_values.to(device), weather_coords.to(device)]
                y = y.to(device)
                #print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                
                optimizer.zero_grad()
                
                y_hat = model(x, z, w)
                #print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)
                loss = masked_mse(y_hat,
                                  y[:,:,0],
                                  y[:,:,1])
                
                print(f"Train loss: {loss}")
                
                loss.backward()
                optimizer.step()
                
                metrics = {
                    "train_loss" : loss
                }
                wandb.log(metrics)              
                
    end_time = time.time()
    exec_time = end_time-start_time

    wandb.log({"tr_epoch_exec_t" : exec_time})

    model_name = 'model_{}_{}.pt'.format(timestamp, i)    
    model_dir = dict_files["save_model_dir"]
    torch.save(model.state_dict(), f"{model_dir}/{model_name}") 

    print(f"############### Test epoch {i} ###############")
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    start_time = time.time()
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
                for batch_idx, (x, z, w_values, y) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")

                    x = x.to(device)
                    z = z.to(device)
                    w = [w_values.to(device), weather_coords.to(device)]
                    y = y.to(device)
                    # print('Batch mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    y_hat = model(x, z, w)
                    # print('After predict mem allocated in MB: ', torch.cuda.memory_allocated() / 1024**2)

                    loss = masked_mse(y_hat,
                                  y[:,:,0],
                                  y[:,:,1])
                    print(f"Test loss: {loss}")

                    metrics = {
                        "test_loss" : loss
                    }

                    wandb.log(metrics)
        
    end_time = time.time()
    exec_time = end_time-start_time
    wandb.log({"test_epoch_exec_t" : exec_time})

wandb.finish()

print(f"Execution time: {end_time-start_time}s")


