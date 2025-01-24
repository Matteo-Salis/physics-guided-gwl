import numpy as np
import torch

from loss.losses_1d import *
from plot.prediction_plot_1d import *

def feedforward_dloss(model, input, groundtruth, loss_prefix_name, require_grad = True):
        
        torch.set_grad_enabled(require_grad)
        pred_hat = model(x = input[0],
                      z = input[1],
                      w = input[2],
                      x_mask = input[3])
        
        loss = masked_mse(pred_hat, groundtruth[0], groundtruth[1])
        loss_dict = {f"{loss_prefix_name}_loss_data": loss}
        return loss_dict
    
    
    
    
def feedforward_dloss_pdeloss(model, input, groundtruth,
                        g,
                        k_lat,
                        k_lon,
                        S_y,
                        lon_cpoints,
                        ds,
                        device,
                        coeff_loss_data,
                        coeff_loss_pde,
                        loss_prefix_name,
                        require_grad = True):
        
        
        loss_dict = feedforward_dloss(model, input, groundtruth, loss_prefix_name, require_grad)
        if require_grad is False:
                torch.set_grad_enabled(True)
        
        # control point generation
        
        # take one point at random in the batch
        sample_idx = torch.randint(0,input[0].shape[0],
                                                 (1,))
        
        x = input[0][sample_idx,:,:]#.unsqueeze(0)
        x_mask = input[3][sample_idx,:]#.unsqueeze(0)
        
        w = [input[2][0][sample_idx,:,:,:,:],#.unsqueeze(0).to(device),
        input[2][1][sample_idx,:,:,:]]#.unsqueeze(0).to(device)]
        
        lat_cpoints = int(lon_cpoints * 0.75) if lon_cpoints > 1 else 1
        total_cpoint = lat_cpoints * lon_cpoints
        x_cpoint = x.expand(total_cpoint,-1,-1).to(device)
        x_mask_cpoint = x_mask.expand(total_cpoint,-1).to(device)
        
        z_cpoint = cpoint_generation(minX = ds.dtm_roi.x.min().values, maxX = ds.dtm_roi.x.max().values,
                                                minY = ds.dtm_roi.y.min().values, maxY = ds.dtm_roi.y.max().values,
                                                dtm = (ds.dtm_roi *ds.norm_factors["dtm_std"]) + ds.norm_factors["dtm_mean"],
                                                num_lon_point = lon_cpoints,
                                                num_lat_point = lat_cpoints)
        
        # normalization 
        z_cpoint[:,0] = (z_cpoint[:,0] - ds.norm_factors["lat_mean"])/ds.norm_factors["lat_std"]
        z_cpoint[:,1] = (z_cpoint[:,1] - ds.norm_factors["lon_mean"])/ds.norm_factors["lon_std"]
        z_cpoint[:,2] = (z_cpoint[:,2] - ds.norm_factors["dtm_mean"].values)/ds.norm_factors["dtm_std"].values
                        
        z_cpoint = torch.tensor(z_cpoint, requires_grad=True).to(torch.float32).to(device)
        w_cpoint = [w[0].expand(total_cpoint,-1,-1,-1,-1),
                w[1].expand(total_cpoint,-1,-1,-1)]
        
        pred_hat_cpoints = model(x_cpoint, z_cpoint, w_cpoint, x_mask_cpoint)
        
        loss_pde = physics_loss(pred_hat_cpoints, z_cpoint,
                    g = g,
                    k_lat = k_lat,
                    k_lon = k_lon,
                    S_y = S_y)
        
        total_loss = coeff_loss_data * loss_dict[f"{loss_prefix_name}_loss_data"]  + coeff_loss_pde * loss_pde
    
        ################ 
        loss_dict[f"{loss_prefix_name}_loss_pde"] = loss_pde
        loss_dict[f"{loss_prefix_name}_tot_loss"] = total_loss
        
        return loss_dict