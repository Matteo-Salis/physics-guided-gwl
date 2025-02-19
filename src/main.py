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
from train_test.load_train import training_model
from train_test.load_test import test_model
from optimizer.load_optimizer import load_optimizer

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
        name = config["run_name"],
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
    print("Dataloaders ready.")

    # load model
    device = (
        config["cuda_device"]
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device: ", device)
    
    print("Loading model...")
    model, model_id = load_model(config)
    model = model.to(device)
    
    wandb.watch(model, log_freq=100)
    print("Model :", model_id)
    print("Total number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = 'model_{}_{}'.format(model_id, timestamp)
    model_dir = config['save_model_dir']
    
    train = training_model(config)
    test = test_model(config)

    print(f"Start time: {timestamp}")
    # optimization parameters
    optimizer = load_optimizer(config, model)
    
    

    # loop training and test
    for epoch in range(config["epochs"]):
        print(f"############### Training epoch {epoch} ###############")
        model.train(True)
        start_time = time.time()

        train(epoch = epoch, dataset = dataset, model = model, train_loader = train_loader,
              optimizer = optimizer, model_dir = model_dir, model_name = model_name, device = device)

        end_time = time.time()
        exec_time = end_time-start_time
        wandb.log({"tr_epoch_exec_t" : exec_time})

        # saving model
        torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")

        print(f"############### Test epoch {epoch} ###############")
        model.eval()
        start_time = time.time()

        test(epoch = epoch, dataset = dataset, model = model, test_loader = test_loader,
             device = device)

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