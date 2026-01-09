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
    
    # Filenames lists
    csv_names = ["_TS_true.csv",
                 "_TS_pred.csv",
                 "_TS_DeltaGW.csv",
                 "_TS_R.csv",
                 "_TS_D.csv",
                 "_TS_lagGWL.csv"]
    
    xr_names = ["_GWLxr.nc",
                "_WTDxr.nc",
                "_DeltaGWxr.nc",
                "_Rxr.nc",
                "_Dxr.nc",
                "_lagGWLxr.nc"]

    dataset = dataset_ST_MultiPoint.Dataset_ST_MultiPoint(config)
    
    models_predictions = {}
    
    print("Loading predictions...", end = " ")
    # Load predictions
    for i in range(len(config["model_name"])):
        
        # load only available csv
        ts_ds_list = []
        xr_ds_list = []
        if config["get_displacement"] and config["model_name"][i] in config["model_with_displacements"]:
            break_id = None
        else:
            break_id = 1
            
        for j in range(len(csv_names)):
            
            if config["n_pred_ts"]>0:
                ts_ds_list.append(pd.read_csv(f"{config['prediction_dir']}/predictions/{config['model_name'][i]}{name_suffix}{csv_names[j]}",
                                        index_col=0, parse_dates=True, dtype=np.float32))
            else:
                ts_ds_list.append(None)
            
            if config["n_pred_map"]>0:
                xr_ds_list.append(xarray.load_dataarray(f"{config['prediction_dir']}/predictions/{config['model_name'][i]}{name_suffix}{xr_names[j]}"))
            else:
                xr_ds_list.append(None)
            
            if j == break_id:
                break
            
        models_predictions[config["model_name"][i]] = [ts_ds_list, xr_ds_list]
        
    print("Done!")
            
    if config["n_pred_ts"]>0:
    # plot ts?
    
        # Create Saving Directory
        ts_saving_path = config["prediction_dir"]+"/time_series"
        os.makedirs(ts_saving_path)
            
        # Time Series plot
        print("Drawing plots...")
        if config["recon_ts"] is True:
            # set params for whole ts plot
            markersize = 1.5
            markersize_true_data = markersize
            linewidth = 0.2
            date_xticks = pd.date_range(np.datetime64("2001-01-01"),
                                        np.datetime64("2023-12-31"),
                                        freq = "YS",  normalize = True,
                                        inclusive = "both")
            date_xticks_format = '%m/%Y'
        else:
            # set params for test set ts plot
            markersize = 2.5
            markersize_true_data = markersize + 1
            linewidth = 0.8
            date_xticks = None
            #date_xticks_format = '%d/%m/%Y'
            
        for sensor_idx in range(len(dataset.sensor_id_list)):

            sensor = dataset.sensor_id_list[sensor_idx]
            munic = dataset.wtd_geodf.loc[dataset.wtd_geodf["sensor_id"] == sensor,"munic"].values[0]

            plt.rcParams.update({'font.size': 16})
            fig, ax = plt.subplots(1,1, figsize = (13,3)) #(12,5)
            
            plt.title(f"{munic} - {sensor}")
            
            markers = ['s', 'D', '^', 'v', '<', '>', 'P', '*', 'X', 'd', 'H', '|', '_']
            colors = config["ts_colors"] #['tab:brown','tab:orange','darkgreen','darkmagenta']
            i = 0
            for model_i in config["model_name"]:
                
                if config["forecast_horizon"] is not None:
                    for j in range(config["n_pred_ts"]):
                        models_predictions[model_i][0][1][sensor].iloc[j*config["forecast_horizon"]:(j+1)*config["forecast_horizon"]].plot(
                                                                                            ax = ax,
                                                                                            color = colors[i % len(markers)],
                                                                                            marker=markers[i % len(markers)],
                                                                                            label = f"{model_i}" if j == config["n_pred_ts"]-1 else "",
                                                                                            markersize = markersize,
                                                                                            linewidth = linewidth
                                                                                            )
                else:
                    models_predictions[model_i][0][1][sensor].plot(label = f"{model_i}", ax = ax,
                                                            color = colors[i % len(markers)],
                                                            marker=markers[i % len(markers)],
                                                            markersize = markersize,
                                                            linewidth = linewidth
                                                            )
                
                if model_i == config["model_name"][-1] :
                    models_predictions[model_i][0][0][sensor].plot(label = "Truth", ax = ax,
                                                                color = "tab:blue",
                                                                marker = "o", linestyle = "--" ,
                                                                markersize = markersize_true_data,
                                                                linewidth = linewidth
                                                                )
                    
                
                i += 1
                
            ax.set_ylim([ax.get_ylim()[0] - ax.get_ylim()[0]*0.0005,
                    ax.get_ylim()[1] + ax.get_ylim()[1].min()*0.0005])
            
            # Start Test Vline
            ax.vlines(config["test_split_p"], ymin = ax.get_ylim()[0],
                    ymax = ax.get_ylim()[1], ls = "--", color = "darkred", lw = 2,
                    label = "Start Test")
            
            # Grey boxes for missing values
            all_dates = models_predictions[model_i][0][0][sensor].index.get_level_values(0)
            if (models_predictions[model_i][0][0][sensor].isnull().any()):
                ax.bar(all_dates[models_predictions[model_i][0][0][sensor].isnull()],
                        bottom = ax.get_ylim()[0],
                        height = ax.get_ylim()[1],
                        width= 2,
                        align='center',
                        color = 'lightgrey',
                        label = "Missing Values", zorder = 0)
            
            print(f"Saving Time Series of {munic} - {sensor}")
            plt.xlabel("Date")
            plt.ylabel("Groundwater Level [m]")
            plt.legend(ncol=np.ceil(len(plt.gca().get_legend_handles_labels()[0])/2),
                       fontsize=12, markerscale=1.5, borderpad=0.2, labelspacing=0.1)
            ax.grid(axis="x", ls = "--", which = "both", lw = "1.5", color = 'black', alpha = 0.5)
            
            if date_xticks is not None:
                ax.set_xticks(date_xticks, date_xticks.strftime(date_xticks_format))
                ax.tick_params(axis = "x", rotation=25)
            
            if config["forecast_horizon"] is None:
                n_pred = config['n_pred_ts']
            else:
                n_pred = config['n_pred_ts']*config["forecast_horizon"]
            title = f"{ts_saving_path}/{munic}_{sensor}_{config['start_date_pred_ts']}_{n_pred}"
            
            if config["iter_pred"]:
                title += "_iter_pred"
                
            if config["forecast_horizon"] is not None:
                title += f"_FO{config['forecast_horizon']}"
                
            plt.savefig(f"{title}.png", bbox_inches='tight', dpi=600, pad_inches=0.1) #dpi = 400, transparent = True
            plt.close("all")
                
                
        print("All time series plots saved!")
    
    # Map plots
    if config["n_pred_map"]>0:
        
        # Create Saving Directory
        map_saving_path = config["prediction_dir"]+"/maps"
        os.makedirs(map_saving_path)
        
        print("Drawing maps...", end = " ")
        for date in config["map_dates"]:
            
            save_map_dir = f"{map_saving_path}/maps_{date.replace('-','_')}"
            
            if config["iter_pred"]:
                save_map_dir += "_iter_pred"
                
            if config["forecast_horizon"] is not None:
                save_map_dir += f"_FO{config['forecast_horizon']}"
            
            ### Map Plots H 
            model_pred_list_H = [models_predictions[config["model_name"][i]][1][0].sel(time = date) for i in range(len(config["model_name"]))]
            model_pred_list_WTD = [models_predictions[config["model_name"][i]][1][1].sel(time = date) for i in range(len(config["model_name"]))]
        
            plot_ST_MultiPoint.plot_map_all_models(model_pred_list_H,
                title = f"{date} Groundwater Level Predictions",
                shapefile = dataset.piemonte_shp,
                model_names = config["model_name"],
                cmap = "Blues",
                var_name_title = "Groundwater Level [m]",
                save_dir = save_map_dir + "_GWL", 
                print_plot = False)
            plt.close("all")
            
            ### Map Plots WTD
            
            plot_ST_MultiPoint.plot_map_all_models(model_pred_list_WTD,
                title = f"{date} Water Table Depth Predictions",
                shapefile = dataset.piemonte_shp,
                model_names = config["model_name"],
                cmap = "Blues_r",
                var_name_title = "Water Table Depth [m]",
                save_dir = save_map_dir + "_WTD", 
                print_plot = False)
            plt.close("all")
            ### Map Plots Displacements
            model_pred_displacements_list = [] 
            
            for i in range(len(config["model_with_displacements"])):
                for j in range(3):
                    
                    displacement_list = []
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][2].sel(time = date))
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][3].sel(time = date))
                    displacement_list.append(models_predictions[config["model_with_displacements"][i]][1][4].sel(time = date))
                    
                model_pred_displacements_list.append(displacement_list)
            
            plot_ST_MultiPoint.plot_displacement_all_models(model_pred_displacements_list,
                title = f"{date} Predicted Equation Components",
                shapefile = dataset.piemonte_shp,
                recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                model_names = config["model_with_displacements"],
                save_dir = save_map_dir + "_Disp", 
                print_plot = False)
            plt.close("all")
        
        print("All Maps saved!")
        #######
        # Gif #
        #######
        
        print("Drawing GIFs...", end = " ")
        save_gif_dir = f"{map_saving_path}/gif_from_{config['start_date_pred_map'].replace('-','_')}"
            
        if config["iter_pred"]:
            save_gif_dir += "_iter_pred"
            
        if config["forecast_horizon"] is not None:
            save_gif_dir += f"_FO{config['forecast_horizon']}"
        
        ### H
        for model in config["model_name"]:
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][0],
                            title = f"{model} - Groundwater Level [m] Evolution",
                            shapefile = dataset.piemonte_shp,
                            freq = "W",
                            cmap = "Blues",
                            vmin_1 = False,
                            vmax_1 = False,
                            save_dir = save_gif_dir + f"_GWL_{model}",
                            print_plot = False)
            
            plt.close("all")
            
        
        for model in config["model_with_displacements"]:
            ### Delta GW
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][2],
                            title = r"{} $\hat{{\Delta}}_{{GW_{{t^*}}}}$ [m] Evolution".format(model),
                            shapefile = dataset.piemonte_shp,
                            recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                            freq = "W",
                            cmap = "seismic_r",
                            save_dir = save_gif_dir + f"_DGW_{model}",
                            print_plot = False)
            plt.close("all")
        
            ### Delta R
            plot_ST_MultiPoint.generate_gif_from_xr(config['start_date_pred_map'], config["n_pred_map"],
                            models_predictions[model][1][3],
                            title = r"{} $\hat{{\mathcal{{R}}}}_{{t^*}}$ [m] Evolution".format(model),
                            shapefile = dataset.piemonte_shp,
                            recharge_areas = dataset.recharge_area_buffer_shp if config["plot_recharge_areas"] else None,
                            freq = "W",
                            cmap = "seismic_r",
                            save_dir = save_gif_dir + f"_R_{model}",
                            print_plot = False)
            plt.close("all")
            
            print("All GIFs saved!")
      

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