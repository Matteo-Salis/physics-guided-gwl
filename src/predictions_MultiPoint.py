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
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_names_dir = "_".join(config["model_name"])
    save_dir = '{}/{}_{}'.format(config["save_dir"],model_names_dir,timestamp)
    
    # Create Saving Directories
    os.makedirs(save_dir)
    ts_saving_path = save_dir+"/time_series"
    map_saving_path = save_dir+"/maps"
    metrics_saving_path = save_dir+"/metrics"
    os.makedirs(ts_saving_path)
    os.makedirs(map_saving_path)
    os.makedirs(metrics_saving_path)


    dataset = dataset_ST_MultiPoint.Dataset_ST_MultiPoint(config)
    
    models_predictions = {}
    
    Z_grid = plot_ST_MultiPoint.grid_generation(dataset,
                                                config["lat_lon_npoints"][0],
                                                config["lat_lon_npoints"][1])
    
    # Load the model and compute predictions
    for i in range(len(config["model_name"])):
        
        model_config = {}
        with open(config["model_config_path"][i]) as f:
            model_config = json.load(f)
            print(f"Read data.json: {config['model_config_path'][i]}")
        
        model = load_model(model_config, config["model_path"][i])
        model = model.to(device)
        print("Total number of trainable parameters: " ,sum(p.numel() for p in model.parameters() if p.requires_grad and p != "Densification_Dropout"))
        
        if config["get_displacement"] and config["model_name"][i] in config["model_with_displacements"]:
            
            models_predictions[config["model_name"][i]] = compute_prediction_with_displacement(config, dataset, device,
                                                                                            model,
                                                                                            config["iter_pred"],
                                                                                            Z_grid)
            
        else:
            
            models_predictions[config["model_name"][i]] = compute_prediction(config, dataset, device,
                                                                            model,
                                                                            config["iter_pred"],
                                                                            Z_grid)
            
      

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