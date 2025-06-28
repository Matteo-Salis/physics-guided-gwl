import numpy as np
import torch
import torch.nn.functional as F


def loss_masked_mse(Y_hat, Y, Y_mask, input):
    
    if input == "ViViT_STMoE":
        if len(Y_hat.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
    elif input == "SparseData_STMoE":
        if len(Y_hat.size()) < 3:
            Y_hat = Y_hat.unsqueeze(0)      
    elif input == "Spatial_STMoE" or input == "Spatial_STMoE_Light":
        if len(Y_hat.size()) < 2:
            Y_hat = Y_hat.unsqueeze(0)
    
    if torch.sum(Y_mask) != 0:
        return torch.sum((Y_hat[Y_mask]-Y[Y_mask])**2.0)  / torch.sum(Y_mask)
    else:
        return 0.

def loss_masked_nse(Y_hat, Y, Y_mask, input):
    
    if input == "ViViT_STMoE":
        if len(Y_hat.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
            
        pass
    
    elif input == "SparseData_STMoE":
        if len(Y_hat.size()) < 3:
            Y_hat = Y_hat.unsqueeze(0)      
        
        sensor_loss = []
        
        avail_sensor_mask = torch.sum(Y_mask, dim = 1)
        
        
        for sensor_idx in range(Y.shape[2]):
            
            sensor_true = Y[:,:,sensor_idx]
            sensor_pred = Y_hat[:,:,sensor_idx]
            sensor_mask = Y_mask[:,:,sensor_idx]
            
            if torch.sum(sensor_mask) != 0:
            
                sensor_true_filled = torch.where(sensor_mask, sensor_true, sensor_pred)
                residuals = torch.sum((sensor_pred - sensor_true_filled)**2, dim = 1)
                Y_mean = torch.mean(sensor_true, dim = 1)
                Y_variance = torch.nansum((Y_filled))
                
                return torch.sum((Y_hat[Y_mask]-Y[Y_mask])**2.0)  / torch.sum(Y_mask)
                sensor_loss.append()
            else:
                sensor_loss.aapend(0)
        
    elif input == "Spatial_STMoE" or input == "Spatial_STMoE_Light":
        if len(Y_hat.size()) < 2:
            Y_hat = Y_hat.unsqueeze(0)
            
        pass
    
    

def loss_masked_mae(Y_hat, Y, Y_mask, input):
    
    if input == "ViViT_STMoE":
        if len(Y_hat.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
    elif input == "SparseData_STMoE":
        if len(Y_hat.size()) < 3:
            Y_hat = Y_hat.unsqueeze(0)
    elif input == "Spatial_STMoE" or input == "Spatial_STMoE_Light":
        if len(Y_hat.size()) < 2:
            Y_hat = Y_hat.unsqueeze(0)

    if torch.sum(Y_mask) != 0:        
        return torch.sum(torch.abs(Y_hat[Y_mask]-Y[Y_mask]))  / torch.sum(Y_mask)
    else:
        return 0

def loss_l2_regularization(model):
    
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l2_norm

def loss_masked_focal_mse(Y_hat, Y, Y_mask, offset_perc = 0):
        """
        
        """
        # if len(Y_hat.size()) < 4:
        #     Y_hat = Y_hat.unsqueeze(0)
        
        offset = round(Y_hat.shape[1] * offset_perc)
        focal_weights = torch.arange(Y_hat.shape[1]+offset,offset,-1).to(Y_hat.device)/(Y_hat.shape[1] + offset)
        
        focal_weights = focal_weights[None,:,None,None].expand(Y_hat.shape[0], -1, Y_hat.shape[2], Y_hat.shape[3])
        focal_weights = torch.where(Y_mask, focal_weights, 0)
        
        Y_filled = torch.where(Y_mask, Y, Y_hat)
        squared_errors = (Y_hat-Y_filled)**2.0
        
        focal_weights_sum = torch.sum(focal_weights, dim = 1)
        lead_time_mse = torch.sum(squared_errors * focal_weights, dim = 1)/(focal_weights_sum+1e-7)
        
        spatial_mask = focal_weights_sum > 0 
        total_mse = torch.nanmean(lead_time_mse[spatial_mask])
        return total_mse

def loss_masked_focal_mae(Y_hat, Y, Y_mask, offset_perc = 0):
        """
        
        """
        # if len(Y_hat.size()) < 4:
        #     Y_hat = Y_hat.unsqueeze(0)
        
        offset = round(Y_hat.shape[1] * offset_perc)
        focal_weights = torch.arange(Y_hat.shape[1]+offset,offset,-1).to(Y_hat.device)/(Y_hat.shape[1] + offset)
        
        focal_weights = focal_weights[None,:,None,None].expand(Y_hat.shape[0], -1, Y_hat.shape[2], Y_hat.shape[3])
        focal_weights = torch.where(Y_mask, focal_weights, 0)
        
        Y_filled = torch.where(Y_mask, Y, Y_hat)
        squared_errors = torch.abs(Y_hat-Y_filled)
        
        focal_weights_sum = torch.sum(focal_weights, dim = 1)
        lead_time_mse = torch.sum(squared_errors * focal_weights, dim = 1)/(focal_weights_sum+1e-7)
        
        spatial_mask = focal_weights_sum > 0 
        total_mse = torch.nanmean(lead_time_mse[spatial_mask])
        return total_mse


def Fdiff_conv(x, mode = "first_lon"):
    if mode == "first_lon":
        kernel = torch.Tensor([[0.,0.,0.],
                               [0.,-1.,1.],
                               [0.,0.,0.]])
    elif mode == "first_lat":
        kernel = torch.Tensor([[0.,1.,0.],
                               [0.,-1.,0.],
                               [0.,0.,0.]])
        
    elif mode == "first_all":
        kernel = torch.Tensor([[1.,1.,1.],
                               [1.,-8.,1.],
                               [1.,1.,1.]])
        
    elif mode == "second":
        
        # TODO
        pass
        # kernel = torch.Tensor([[0.,0.,0.],
        #                        [-1,0,1],
        #                        [0.,0.,0.]])
        
    kernel = kernel.view(1,1,3,3).to(x.device) #(out_channels, in_channels, kH, KW)
    
    # Padding 
    # (padding_left, padding_right, padding_top, padding_bottom)
    x_padded = F.pad(x, pad = (1,1,1,1), mode = "replicate")
    
    output = F.conv2d(x_padded, kernel, padding = "valid")
    
    return output

def physics_loss(Y_hat, dataset, K_lat = 1., K_lon = 1., G = 0.,
                 loss = "mae"):
    
    #Y_hat_denorm = (Y_hat * dataset.norm_factors["target_std"]) + dataset.norm_factors["target_mean"]
    
    std = torch.from_numpy(dataset.target_stds_xr.values).to(Y_hat.device).to(torch.float32)
    mean = torch.from_numpy(dataset.target_means_xr.values).to(Y_hat.device).to(torch.float32)
    
    Y_hat_denorm = (Y_hat * std) + mean
    
    
    spatial_grads = []
    
    for t in range(Y_hat_denorm.shape[1]-1):
    
        Y_hat_t = Y_hat_denorm[:,t,:,:].unsqueeze(1)
        dh_dy = Fdiff_conv(Y_hat_t, mode = "first_lat")
        dh_dx = Fdiff_conv(Y_hat_t, mode = "first_lon")
        
        dh_dy = dh_dy * K_lat
        dh_dx = dh_dx * K_lon
        
        dh_dydy = Fdiff_conv(dh_dy, mode = "first_lat")
        dh_dxdx = Fdiff_conv(dh_dx, mode = "first_lon")
    
        spatial_grad = dh_dydy + dh_dxdx
        
        spatial_grads.append(spatial_grad)
        
    spatial_grads = torch.cat(spatial_grads, dim = 1).to(Y_hat_denorm.device)
    
    temporal_grad = Y_hat_denorm[:,1:,:,:] - Y_hat_denorm[:,:-1,:,:]
    
    residuals = temporal_grad - spatial_grads - G
    
    residuals_norm = residuals / std
    
    if loss == "mae":
        phyiscs_loss = torch.mean(torch.abs(residuals_norm))
    elif loss == "mse":
        phyiscs_loss = torch.mean(residuals_norm**2)
        
    return phyiscs_loss

if __name__ == "__main__":
    pass