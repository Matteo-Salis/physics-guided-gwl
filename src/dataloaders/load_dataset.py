import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler

# from dataloaders.load_1d_meteo_wtd import ContinuousDataset
from dataloaders.load_2d_meteo_wtd import DiscreteDataset


def load_dataset(config):
    
    if config["dataset_type"] == "discrete":
        return DiscreteDataset(config)
    # elif config["dataset_type"] == "continuous":
    #     return ContinuousDataset(config)
    else:
        raise Exception("Model name unknown.")
    

def get_dataloader(dataset, config):

    test_split_p = config["test_split_p"]
    train_split_p = 1 - test_split_p
    
    max_ds_elems = dataset.__len__()
    if not config["all_dataset"]:
        max_ds_elems = config["max_ds_elems"]

    train_idx = int(max_ds_elems*train_split_p)
    test_idx = int(max_ds_elems*test_split_p)

    print(f"Traing size: {train_idx}, Test size: {test_idx}")

    train_idxs, test_idxs = np.arange(train_idx), np.arange(train_idx, train_idx + test_idx)

    train_sampler = RandomSampler(train_idxs)
    test_sampler = RandomSampler(test_idxs)

    if config["sampler"] == "SequentialSampler":
        train_sampler = SequentialSampler(train_idxs)
        test_sampler = SequentialSampler(test_idxs)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                sampler=test_sampler)
    
    return train_loader, test_loader