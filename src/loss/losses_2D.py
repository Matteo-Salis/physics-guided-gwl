import numpy as np
import torch
import torch.nn.functional as F


def loss_masked_mse(Y_hat, Y, Y_mask):
    
    if len(Y_hat.size()) < 4:
        Y_hat = Y_hat.unsqueeze(0)
        
    return torch.sum((Y_hat[Y_mask]-Y[Y_mask])**2.0)  / torch.sum(Y_mask)

def loss_l2_regularization(model):
    
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l2_norm

def loss_masked_focal_mse(Y_hat, Y, Y_mask, offset_perc = 0):
        """
        
        """
        if len(Y_hat.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
        
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
        if len(Y_hat.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
        
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

def physics_loss(Y_hat, K_lat = 1., K_lon = 1., G = 0.):
    
    spatial_grads = []
    
    for t in range(Y_hat.shape[1]-1):
    
        Y_hat_t = Y_hat[:,t,:,:].unsqueeze(1)
        dh_dy = Fdiff_conv(Y_hat_t, mode = "first_lat")
        dh_dx = Fdiff_conv(Y_hat_t, mode = "first_lon")
        
        dh_dy = dh_dy * K_lat
        dh_dx = dh_dx * K_lon
        
        dh_dydy = Fdiff_conv(dh_dy, mode = "first_lat")
        dh_dxdx = Fdiff_conv(dh_dx, mode = "first_lon")
    
        spatial_grad = dh_dydy + dh_dxdx
        
        spatial_grads.append(spatial_grad)
        
    spatial_grads = torch.cat(spatial_grads, dim = 1).to(Y_hat.device)
    
    temporal_grad = Y_hat[:,1:,:,:] - Y_hat[:,:-1,:,:]
    
    residuals = temporal_grad - spatial_grads - G
    
    residuals = torch.mean(residuals**2)
        
    return residuals

if __name__ == "__main__":
    pass