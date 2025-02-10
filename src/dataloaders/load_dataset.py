import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataloaders.dataset_1d import Dataset_1D
from dataloaders.load_2d_meteo_wtd import DiscreteDataset


def load_dataset(config):
    
    if config["dataset_type"] == "discrete":
        return DiscreteDataset(config)
    elif config["dataset_type"] == "1d":
        return Dataset_1D(config)
    else:
        raise Exception("Model name unknown.")
    

def get_dataloader(dataset, config):

    max_ds_elems = dataset.__len__()
    if not config["all_dataset"]:
        max_ds_elems = config["max_ds_elems"]
        
    if type(config["test_split_p"]) is str:
        
        train_idx = int(dataset.get_iloc_from_date(date_max= np.datetime64(config["test_split_p"])))
        test_idx = int(max_ds_elems - train_idx)
    else:
        test_split_p = config["test_split_p"]
        train_split_p = 1 - test_split_p
        
        train_idx = int(max_ds_elems*train_split_p)
        test_idx = int(max_ds_elems*test_split_p)

    train_idxs, test_idxs = np.arange(train_idx), np.arange(train_idx,
                                                            train_idx + test_idx)

    # Print info 
    if config["dataset_type"] == "discrete":
        print(f"Traing size: {train_idx}, Test size: {test_idx}")
    elif config["dataset_type"] == "1d":
        print(f"Traing size: {train_idx} - {dataset.wtd_df.index.get_level_values(0)[train_idxs[-1]]}, Test size: {test_idx} - {dataset.wtd_df.index.get_level_values(0)[test_idxs[-1]]}")

    # Sampler 
    if config["random_sampler"] is True:
        train_sampler = RandomSampler(train_idxs)
    else:
        train_sampler = SequentialSampler(train_idxs)
        
    test_sampler = SequentialSampler(test_idxs)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                sampler=test_sampler)
    
    return train_loader, test_loader


