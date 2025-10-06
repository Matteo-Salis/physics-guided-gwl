from loss.losses_SparseData import *
from functools import partial

def load_loss(config, dataset):
    
    norm_factor = [torch.tensor(dataset.norm_factors["target_means"]).to(torch.float32),
            torch.tensor(dataset.norm_factors["target_stds"]).to(torch.float32)]
    
    if config["loss"] == "mse":
        return partial(loss_masked_mse)
    
    if config["loss"] == "mae":
        return partial(loss_masked_mae)
    
    if config["loss"] == "mape":
        return partial(loss_masked_mape)
    
    if config["loss"] == "h2":
        return partial(loss_masked_h2,
                       norm_factor = norm_factor)
    
    if config["loss"] == "nse":
        return partial(loss_masked_nse,
                       normalized = False)
    
    if config["loss"] == "nnse":
        return partial(loss_masked_nse,
                       normalized = True)

    if config["loss"] == "focal-mse":
        return loss_masked_focal_mse
    
    if config["loss"] == "focal-mae":
        return loss_masked_focal_mae