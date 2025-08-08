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
        return torch.tensor(0.).to(Y_mask.device)

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
        return torch.tensor(0.).to(Y_mask.device)
    
def loss_masked_mape(Y_hat, Y, Y_mask):
    
    if len(Y_hat.shape)==1:
        Y_hat = Y_hat.unsqueeze(0)
        
    not_Y_mask = ~Y_mask
    
    if torch.sum(Y_mask) != 0:
        mape = torch.mean(torch.abs((Y_hat[not_Y_mask]-Y[not_Y_mask])/Y[not_Y_mask]))
    else:
        mape = torch.tensor(0.).to(Y_mask.device)
    return mape
    
    
######################
### Physics Losses ###
######################

##VEDI DeBenzac e optical flow
def displacement_reg(displacement_term,
                     res_fn):
    
    if res_fn == "mse":
        return torch.mean(displacement_term**2)
    
    elif res_fn == "mae" or res_fn == "mape":
        return torch.mean(torch.abs(displacement_term))
    

def smoothness_reg(prediction,
                    mode,
                    step = 1):
    
    if mode == "lon_lat":
        spatial_grads_lat = []
        spatial_grads_lon = []
        for t in range(prediction.shape[0]):
            prediction_t = prediction[t,:,:][None,None,:,:]

            lat_1derivative = Fdiff_conv(prediction_t,
                                    mode = "centered_lat",
                                    der_order = 1)
            lat_1derivative = lat_1derivative / step
            
            lon_1derivative = Fdiff_conv(prediction_t,
                                    mode = "centered_lon",
                                    der_order = 1)
            lon_1derivative = lon_1derivative / step
            
            spatial_grads_lat.append(lat_1derivative)
            spatial_grads_lon.append(lon_1derivative)
            
        spatial_grads_lat = torch.stack(spatial_grads_lat, dim = 0).to(prediction.device).squeeze()
        spatial_grads_lon = torch.stack(spatial_grads_lon, dim = 0).to(prediction.device).squeeze()
        
        loss_lat = torch.mean(spatial_grads_lat**2)
        loss_lon = torch.mean(spatial_grads_lon**2)
        
        return [loss_lat, loss_lon]
            
    elif mode == "temp":
        
        temp_1derivative = torch.diff(prediction, n=1, dim=0) 
    
        return torch.mean(temp_1derivative**2)


def coherence_loss(Lag_GW_true,
                Lag_GW_true_mask,
                res_fn,
                Lag_GW_hat = None):
    
    # Y_hat, Displacement_GW, and Displacement_S no batch dimension
    not_Lag_GW_true_mask = ~Lag_GW_true_mask
    residuals = Lag_GW_true[not_Lag_GW_true_mask] - Lag_GW_hat[not_Lag_GW_true_mask]
    
    if res_fn == "mse":
        return torch.mean(residuals**2)
    
    elif res_fn == "mae":
        return torch.mean(torch.abs(residuals))
    
    elif res_fn == "mape":
        return torch.mean(torch.abs(residuals/(Lag_GW_true[not_Lag_GW_true_mask] + 1e-7)))

def diffusion_loss(Lag_GW, Displacement_GW, K,
                   normf_mu, normf_sigma,
                   res_fn,
                   dx = 1820,
                   dy = 2586):
    
    Lag_GW_denorm = (Lag_GW*normf_sigma) + normf_mu
    K_denorm = K*normf_sigma
    Displacement_GW_denorm = Displacement_GW*normf_sigma
    
    spatial_grads = []
    
    for t in range(Lag_GW_denorm.shape[0]):
    
        Lag_GW_t = Lag_GW_denorm[t,:,:][None,None,:,:]
        
        lon_2derivative = Fdiff_conv(Lag_GW_t,
                                mode = "centered_lon",
                                der_order = 2)
        lon_2derivative = lon_2derivative / (dx**2)

        lat_2derivative = Fdiff_conv(Lag_GW_t,
                                mode = "centered_lat",
                                der_order = 2)
        lat_2derivative = lat_2derivative / (dy**2)
        
        spatial_grad = (lon_2derivative + lat_2derivative)*K_denorm[t,:,:][None,None,:,:]
        
        spatial_grads.append(spatial_grad)

    spatial_grads = torch.cat(spatial_grads, dim = 1).to(Lag_GW.device).squeeze()
    residuals = Displacement_GW_denorm - spatial_grads
    residuals = residuals/normf_sigma
        
    if res_fn == "mse":
        return torch.mean(residuals**2)
    
    elif res_fn == "mae":
        return torch.mean(torch.abs(residuals))
    
    elif res_fn == "mape":
        return torch.mean(torch.abs(residuals/((spatial_grads+1e-5)/normf_sigma)))

def Fdiff_conv(x, mode = "first_lon", der_order = 1):
    """
    Finite Difference Approximation using Convolution 
    """
    if der_order == 1:
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
            
        elif mode == "centered_lon":
            
            kernel = torch.Tensor([[0.,0.,0.],
                                [-0.5,0.,0.5],
                                [0.,0.,0.]])
            
        elif mode == "centered_lat":
            
            kernel = torch.Tensor([[0.,0.5,0.],
                                [0.,0.,0.],
                                [0.,-0.5,0.]])
            
            
    elif der_order == 2:
        
        if mode == "centered_lon":
            kernel = torch.Tensor([[0.,0.,0.],
                                [1.,-2.,1.],
                                [0.,0.,0.]])
            
        elif mode == "centered_lat":
            kernel = torch.Tensor([[0.,1.,0.],
                                [0.,-2.,0.],
                                [0.,1.,0.]])
        
        
            
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