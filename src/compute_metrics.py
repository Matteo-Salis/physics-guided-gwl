from operator import itemgetter
from tqdm import tqdm
import time
from datetime import datetime
import json
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray
import rioxarray
import fiona

#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

#from rasterio.enums import Resampling

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
from torch.autograd import Variable
from torchview import draw_graph

import seaborn as sns
import argparse

import sys
import os

### Import Module ###

from dataloaders import dataset_ST_MultiPoint
from models import models_ST_MultiPoint
from models import load_model_ST_MultiPoint
from utils import plot_ST_MultiPoint
from utils import metrics

from utils.test_predictions import load_model
from utils.test_predictions import compute_prediction_with_displacement
from utils.test_predictions import compute_prediction

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--output', default=None, type=str, help='wandb.json file to track runs')
    args = parser.parse_args()
    return args

## Load dataset
def main(config):
    device = (
    config["device"]
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print("Device: ", device)
    
    # Name suffix for filenames
    name_suffix = ""
            
    if config["iter_pred"]:
        name_suffix += "_iter_pred"
        
    if config["forecast_horizon"] is not None:
        name_suffix += f"_FO{config['forecast_horizon']}"
    
    # Create Saving Directory
    metrics_saving_path = config["prediction_dir"]+"/metrics"
    os.makedirs(metrics_saving_path)

    # Filenames lists
    csv_names = ["_TS_true.csv",
                 "_TS_pred.csv",
                 "_TS_DeltaGW.csv",
                 "_TS_R.csv",
                 "_TS_D.csv",
                 "_TS_lagGWL.csv"]
    
    # xr_names = ["_GWLxr.nc",
    #             "_WTDxr.nc",
    #             "_DeltaGWxr.nc",
    #             "_Rxr.nc",
    #             "_Dxr.nc",
    #             "_lagGWLxr.nc"]
    
    dataset = dataset_ST_MultiPoint.Dataset_ST_MultiPoint(config)
    
    models_predictions = {}
    
    print("Loading predictions...", end = " ")
    # Load predictions
    for i in range(len(config["model_name"])):
        
        # load only available csv
        ds_list = []
        if config["get_displacement"] and config["model_name"][i] in config["model_with_displacements"]:
            break_id = None
        else:
            break_id = 1
            
        for j in range(len(csv_names)):
            
            ds_list.append(pd.read_csv(f"{config['prediction_dir']}/predictions/{config['model_name'][i]}{name_suffix}{csv_names[j]}",
                                       index_col=0, parse_dates=True, dtype=np.float32))
            
            if j == break_id:
                break
            
        models_predictions[config["model_name"][i]] = ds_list
        
    print("Done!")
            
    # Compute metrics
        
    print("Computing metrics...")
    median_metrics_dict = {}
    mean_metrics_dict = {}
    std_metrics_dict = {}
    iqr_metrics_dict = {}
        
    #Compute denormalized sensor statistics
    subset_wtd_df = dataset.wtd_df.loc[pd.IndexSlice[dataset.wtd_df.index.get_level_values(0) <= np.datetime64(dataset.config["date_max_norm"]),
                                                    dataset.sensor_id_list_target]] #
    
    subset_wtd_df = (subset_wtd_df[dataset.target] * dataset.norm_factors["target_stds"]) + dataset.norm_factors["target_means"]
    
    sensor_means = subset_wtd_df.groupby(level=1).transform('mean').values
    sensor_means = sensor_means.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list_target),
                                        len(dataset.sensor_id_list_target))[0,:] 
    
    sensor_min = subset_wtd_df.groupby(level=1).transform('min').values
    sensor_min = sensor_min.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list_target),
                                        len(dataset.sensor_id_list_target))[0,:] 
    
    sensor_max = subset_wtd_df.groupby(level=1).transform('max').values
    sensor_max = sensor_max.reshape(len(subset_wtd_df.index)//len(dataset.sensor_id_list_target),
                                        len(dataset.sensor_id_list_target))[0,:] 
    sensor_iv = sensor_max - sensor_min

    # computing metrics    
    for model_i in config["model_name"]:

        model_median_metrics = []
        model_mean_metrics = []
        model_std_metrics = []
        model_iqr_metrics = []
        
        # define temporal domain
        if config["metrics_only_on_test"] is True:
            true_values = models_predictions[model_i][0].loc[models_predictions[model_i][0].index>np.datetime64(config["test_split_p"])]
            predicted_values = models_predictions[model_i][1].loc[models_predictions[model_i][0].index>np.datetime64(config["test_split_p"])]
        else:
            true_values = models_predictions[model_i][0]
            predicted_values = models_predictions[model_i][1]
        
        # compute metrics 
        # nbias    
        sensors_nbias = metrics.compute_test_nbias_per_sensor(true_values,
                                                            predicted_values,
                                                            sensor_iv)
        sensors_nbias.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_nbias.csv", index=True)
        model_median_metrics.append(sensors_nbias.median())
        model_mean_metrics.append(sensors_nbias.mean())
        model_std_metrics.append(sensors_nbias.std())
        model_iqr_metrics.append(sensors_nbias.quantile(0.75)-sensors_nbias.quantile(0.25))
        
        # rmse
        sensors_rmse = metrics.compute_test_rmse_per_sensor(true_values,
                                                            predicted_values)
        sensors_rmse.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_rmse.csv", index=True)
        model_median_metrics.append(sensors_rmse.median())
        model_mean_metrics.append(sensors_rmse.mean())
        model_std_metrics.append(sensors_rmse.std())
        model_iqr_metrics.append(sensors_rmse.quantile(0.75) - sensors_rmse.quantile(0.25))
        
        # ape mape
        sensors_ape = metrics.compute_test_ape_per_sensor(true_values,
                                                            predicted_values)
        sensors_mape = metrics.compute_test_mape_per_sensor(true_values,
                                                            predicted_values)
        sensors_ape.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_ape.csv", index=True)
        sensors_mape.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_mape.csv", index=True)
        model_median_metrics.append(sensors_mape.median())
        model_mean_metrics.append(sensors_mape.mean())
        model_std_metrics.append(sensors_mape.std())
        model_iqr_metrics.append(sensors_mape.quantile(0.75)-sensors_mape.quantile(0.25))
        
        # nse
        sensors_nse = metrics.compute_test_nse_per_sensor(true_values,
                                                        predicted_values,
                                                        sensor_means)
        sensors_nse.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_nse.csv", index=True)
        model_median_metrics.append(sensors_nse.median())
        model_mean_metrics.append(sensors_nse.mean())
        model_std_metrics.append(sensors_nse.std())
        model_iqr_metrics.append(sensors_nse.quantile(0.75)-sensors_nse.quantile(0.25))
        
        # kge
        sensors_kge = metrics.compute_test_kge_per_sensor(true_values,
                                                        predicted_values)
        sensors_kge.to_csv(f"{metrics_saving_path}/{model_i}{name_suffix}_kge.csv", index=True)
        model_median_metrics.append(sensors_kge.median())
        model_mean_metrics.append(sensors_kge.mean())
        model_std_metrics.append(sensors_kge.std())
        model_iqr_metrics.append(sensors_kge.quantile(0.75)-sensors_kge.quantile(0.25))
        
        # filling dictionaries
        median_metrics_dict[model_i] = model_median_metrics
        mean_metrics_dict[model_i] = model_mean_metrics
        std_metrics_dict[model_i] = model_std_metrics
        iqr_metrics_dict[model_i] = model_iqr_metrics

    # dataframe and savings         
    median_metrics_ds = pd.DataFrame(median_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
    mean_metrics_ds = pd.DataFrame(mean_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
    std_metrics_ds = pd.DataFrame(std_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
    iqr_metrics_ds = pd.DataFrame(iqr_metrics_dict, index = ["NBIAS","RMSE","MAPE","NSE","KGE"])
        
    median_metrics_ds.to_csv(f"{metrics_saving_path}/median_metrics{name_suffix}.csv", index=True)
    mean_metrics_ds.to_csv(f"{metrics_saving_path}/mean_metrics{name_suffix}.csv", index=True)
    std_metrics_ds.to_csv(f"{metrics_saving_path}/std_metrics{name_suffix}.csv", index=True)
    iqr_metrics_ds.to_csv(f"{metrics_saving_path}/iqr_metrics{name_suffix}.csv", index=True)
    
    print("################# All metrics saved! #################")
        

if __name__ == "__main__":
    args = parse_arguments()

    config = {}
    with open(args.config) as f:
        config = json.load(f)
    
    if config["stdout_log_dir"] is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_names_dir = "_".join(config["model_name"])
        save_dir_stdout = '{}_{}_{}.txt'.format(config["stdout_log_dir"],model_names_dir,timestamp)
        save_dir_stderr = '{}_{}_{}.txt'.format(config["stderr_log_dir"],model_names_dir,timestamp)
            
        # Redirect sys.stdout and err to the files
        sys.stdout = open(save_dir_stdout, 'w')
        sys.stderr = open(save_dir_stderr, 'w')

    main(config)