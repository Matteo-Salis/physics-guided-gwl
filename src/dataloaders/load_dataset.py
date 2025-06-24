import numpy as np
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataloaders.dataset_1d import Dataset_1D
from dataloaders.dataset_2D import Dataset_2D_ImageCond
from dataloaders.dataset_2D import Dataset_2D_VideoCond
from dataloaders.dataset_2d import DiscreteDataset
from dataloaders.dataset_SparseData import Dataset_Sparse


def load_dataset(config):
    
    if config["dataset_type"] == "2d":
        return DiscreteDataset(config)
    elif config["dataset_type"] == "1d":
        return Dataset_1D(config)
    elif config["dataset_type"] == "2D_ImageCond":
        return Dataset_2D_ImageCond(config)
    elif config["dataset_type"] == "2D_VideoCond":
        return Dataset_2D_VideoCond(config)
    elif config["dataset_type"] == "Dataset_Sparse":
        return Dataset_Sparse(config)
    else:
        raise Exception("Model name unknown.")
    

def get_dataloader(dataset, config):

    if config["all_dataset"] is True:
        max_ds_elems = dataset.__len__()
    else:
        max_ds_elems = config["max_ds_elems"]
        
        
    if type(config["test_split_p"]) is str:
        
        train_idx = int(dataset.get_iloc_from_date(date_max = np.datetime64(config["test_split_p"])))
        test_idx = int(max_ds_elems - train_idx)
    else:
        test_split_p = config["test_split_p"]
        train_split_p = 1 - test_split_p
        
        train_idx = int(max_ds_elems*train_split_p)
        test_idx = int(max_ds_elems*test_split_p)

    train_idxs, test_idxs = np.arange(train_idx + 1), np.arange(train_idx + config["twindow"],
                                                            train_idx + test_idx)

    # Print info 
    if config["dataset_type"] == "2d":
        print(f"Traing size: {train_idx}, Test size: {test_idx - config['twindow']}")
    elif config["dataset_type"] == "1d":
        print(f"Traing size: {train_idx} - {dataset.wtd_df.index.get_level_values(0)[train_idxs[-1]].astype('datetime64[D]')}, Test size: {test_idx} - {dataset.wtd_df.index.get_level_values(0)[test_idxs[-1]].astype('datetime64[D]')}")
    elif config["dataset_type"] == "2D_VideoCond":
        print(f"Traing size: {train_idx} - Start: {dataset.wtd_data_raserized.time.values[train_idxs[0]].astype('datetime64[D]')} - End: {dataset.wtd_data_raserized.time.values[train_idxs[-1]].astype('datetime64[D]')};\nTest size: {test_idx} - Start: {dataset.wtd_data_raserized.time.values[test_idxs[0]].astype('datetime64[D]')} - End: {dataset.wtd_data_raserized.time.values[test_idxs[-1]].astype('datetime64[D]')}")
    elif config["dataset_type"] == "Dataset_Sparse":
        print(f"Traing size: {train_idx} - Start: {np.datetime64(dataset.input_dates[train_idxs[0]]).astype('datetime64[D]')} - End: {np.datetime64(dataset.input_dates[train_idxs[-1]]).astype('datetime64[D]')};\nTest size: {test_idx} - Start: {np.datetime64(dataset.input_dates[test_idxs[0]]).astype('datetime64[D]')} - End: {np.datetime64(dataset.input_dates[test_idxs[-1]]).astype('datetime64[D]')}")


    # Subset
    train_set = torch.utils.data.Subset(dataset, train_idxs)
    test_set = torch.utils.data.Subset(dataset, test_idxs)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"],
                                                shuffle=config["random_sampler"])
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"],
                                                shuffle=False)
    
    return train_loader, test_loader


