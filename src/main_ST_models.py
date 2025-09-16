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
import sys

import torch.nn as nn
from tqdm import tqdm
import torch, numpy as np, random

from dataloaders.load_dataset_ST_MultiPoint import load_dataset, get_dataloader
from models.load_model_ST_MultiPoint import load_model
from train_test.load_train_ST_MultiPoint import training_model
from train_test.load_test_ST_MultiPoint import test_model
from optimizer.load_optimizer import load_optimizer, load_lr_scheduler
from loss.load_losses_ST_MultiPoint import load_loss

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
        name = f"{config['model']}_{config['run_name']}",
        dir = config["wandb_dir"],
        config = config,
        mode = config["wandb_mode"],
    )

def main(config):
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # wandb configuration
    wandb_config(config)

    # load dataset
    print("Loading dataset...")
    dataset = load_dataset(config)
    print("Dataset loaded!")
    print(f"Length of the dataset: {dataset.__len__()}")

    # load dataloader
    print("Getting dataloader...")
    train_loader, test_loader = get_dataloader(dataset, config)
    print("Dataloaders ready!")

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
    model_name = '{}_{}_{}'.format(model_id, config["run_name"], timestamp)
    save_dir = f"{config['save_model_dir']}/{model_name}"
    
    os.makedirs(save_dir, exist_ok=False)
    os.makedirs(f"{save_dir}/ts_plots")
    os.makedirs(f"{save_dir}/map_plots")
    
    train = training_model(config)
    test = test_model(config)

    print(f"Start time: {timestamp}")
    # optimization parameters
    optimizer = load_optimizer(config, model)
    lr_scheduler = load_lr_scheduler(config, optimizer)
    loss_fn = load_loss(config)
    
    
    # loop training and test
    for epoch in range(config["epochs"]):
        print(f"############### Training epoch {epoch} ###############")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        model.train(True)
        start_time = time.time()

        train(epoch = epoch, dataset = dataset, model = model, train_loader = train_loader,
              loss_fn = loss_fn, optimizer = optimizer, model_dir = save_dir,
              model_name = model_name,
              device = device
              )

        end_time = time.time()
        exec_time = end_time-start_time
        wandb.log({"tr_epoch_exec_t" : exec_time})

        # saving model
        if (epoch+1) % 25 == 0 or epoch == config["epochs"]-1:
            print("Saving Model...", end = " ")
            torch.save(model.state_dict(), f"{save_dir}/{model_name}.pt")
            print("Done!")

        print(f"############### Test epoch {epoch} ###############")
        model.eval()
        start_time = time.time()

        test(epoch = epoch, dataset = dataset, model = model, test_loader = test_loader,
             model_dir = save_dir, loss_fn = loss_fn, device = device)

        end_time = time.time()
        exec_time = end_time-start_time
        wandb.log({"test_epoch_exec_t" : exec_time})
        
        if lr_scheduler:
            lr_scheduler.step()   

    wandb.finish()
    print(f"Execution ended.")

    # end main

if __name__ == "__main__":
    args = parse_arguments()

    config = {}
    with open(args.config) as f:
        config = json.load(f)
    
    if config["stdout_log_dir"] is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = '{}_{}_{}'.format(config["model"], config["run_name"], timestamp)
            
        # Redirect sys.stdout and err to the files
        sys.stdout = open(f'{config["stdout_log_dir"]}_{run_id}.txt', 'w')
        sys.stderr = open(f'{config["stderr_log_dir"]}_{run_id}.txt', 'w')

    main(config)