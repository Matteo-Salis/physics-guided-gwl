import numpy as np
import torch
import torch.nn.functional as F

def loss_masked_mae(Y_hat, Y, Y_mask):
    
    # batch dim
    if len(Y_hat.shape)==1:
        Y_hat = Y_hat.unsqueeze(0)
        
    not_Y_mask = ~Y_mask

    if torch.sum(Y_mask) != 0:        
        return torch.sum(torch.abs(Y_hat[not_Y_mask]-Y[not_Y_mask]))  / torch.sum(not_Y_mask)
    else:
        return 0

def loss_l2_regularization(model):
    
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    return l2_norm

def loss_masked_mse(Y_hat, Y, Y_mask):
    
    # batch dim
    if len(Y_hat.shape)==1:
        Y_hat = Y_hat.unsqueeze(0)
    
    not_Y_mask = ~Y_mask
    
    if torch.sum(Y_mask) != 0:
        return torch.sum((Y_hat[not_Y_mask]-Y[not_Y_mask])**2.0)  / torch.sum(not_Y_mask)
    else:
        return 0.
    
    
######################
### Physics Losses ###
######################

##VEDI DeBenzac e optical flow
def displacement_reg():
    pass

def HydroConductivity_reg(HC, denorm_sigma = None):
    
    if denorm_sigma is not None:
      HC = HC * denorm_sigma
    
    penalty = torch.relu(-HC).sum()
    
    return penalty

def coherence_loss(Y_hat,
                Displacement_GW,
                Displacement_S):
    
    Y_hat_t1 = Y_hat[:,1:,:].clone()
    Y_hat_t0 = Y_hat[:,:-1,:].clone()
    
    Displacement_GW = Displacement_GW[:,1:,:].clone()
    Displacement_S = Displacement_S[:,1:,:].clone()
    
    Y_hat_diff = Y_hat_t1-Y_hat_t0
    
    residuals = Y_hat_diff - (Displacement_GW + Displacement_S)
    
    return torch.mean(residuals**2)

def diffusion_loss():
    pass

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