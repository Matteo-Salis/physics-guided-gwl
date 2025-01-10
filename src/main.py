import os
import torch
import time
import logging
import utils
import json
import wandb
import argparse
import numpy as np
from datetime import datetime

import torch.nn as nn
from tqdm import tqdm

from dataloaders.load_dataset import load_dataset, get_dataloader
from models.load_model import load_model
from train import train_model
from test import test_model

wandb.login()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--output', default=None, type=str, help='wandb.json file to track runs')
    args = parser.parse_args()
    return args

def wandb_config(config):
    wandb.init(
        entity = config["entity"],
        project = config["experiment_name"],
        dir = config["wandb_dir"],
        config = config,
        mode = "offline",
    )

def main(config):
    # wandb configuration
    wandb_config(config)

    # load dataset
    print("Loading dataset...")
    dataset = load_dataset(config)
    print("Dataset loaded.")
    print(f"Length of the dataset: {dataset.__len__()}")

    # load dataloader
    print("Getting dataloader...")
    train_loader, test_loader = get_dataloader(dataset, config)
    print("Dataloader ready.")

    # load model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading model...")
    model = load_model(config).to(device)
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'model_{}.pt'.format(timestamp) 
    
    wandb.watch(model, log_freq=100)

    # optimization parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    dtm = torch.from_numpy(dataset.dtm_roi_downsampled.values).to(device)
    wtd_mean = dataset.wtd_numpy_mean
    wtd_std = dataset.wtd_numpy_std

    # loop training and test
    for i in range(config["epochs"]):
        print(f"############### Training epoch {i} ###############")
        model.train()
        start_time = time.time()

        train_model(i, model, train_loader, optimizer, wtd_mean, wtd_std, config, model_name, device)

        end_time = time.time()
        exec_time = end_time-start_time
        wandb.log({"tr_epoch_exec_t" : exec_time})

        # saving model
        torch.save(model.state_dict(), f"{config['save_model_dir']}/{model_name}")

        print(f"############### Test epoch {i} ###############")
        model.eval()
        start_time = time.time()

        test_model(i, model, test_loader, wtd_mean, wtd_std, config, device)

        end_time = time.time()
        exec_time = end_time-start_time
        wandb.log({"test_epoch_exec_t" : exec_time})

    wandb.finish()
    print(f"Execution ended.")

    # end main



if __name__ == "__main__":
    args = parse_arguments()

    config = {}
    with open(args.config) as f:
        config = json.load(f)

    main(config)