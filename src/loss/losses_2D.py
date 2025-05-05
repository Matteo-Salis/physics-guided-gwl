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
        if len(Y.size()) < 4:
            Y_hat = Y_hat.unsqueeze(0)
        
        offset = round(Y.shape[1] * offset_perc)
        focal_weights = torch.arange(Y.shape[1]+offset,offset,-1).to(Y.device)/(Y.shape[1] + offset)
        
        focal_weights = focal_weights[None,:,None, None].expand(Y.shape[0], -1, Y.shape[2], Y.shape[3])
        squared_errors = ((Y_hat-Y)**2.0)
        
        squared_errors = torch.where(Y_mask, squared_errors, torch.nan)
        focal_weights = torch.where(Y_mask, focal_weights, torch.nan)
        
        lead_time_mse = torch.nansum(squared_errors * focal_weights, dim = 1)/torch.nansum(focal_weights, dim = 1)
        total_mse = torch.nanmean(lead_time_mse)
        return total_mse
    
    # true_pred = y[:,:,:-1]
    #     if len(true_pred.shape) == 2:
    #         true_pred = true_pred.unsqueeze(1)
    #     mask = y[:,:,-1].unsqueeze(-1).expand(-1,-1,y_hat.shape[-1])
    #     mask = mask.to(torch.bool)
        
    #     if loss_focal_weights is True:
    #         offset = round(true_pred.shape[1] * offset_perc)
    #         focal_weights = torch.arange(true_pred.shape[1]+offset,offset,-1).to(true_pred.device)/(true_pred.shape[1] + offset)
    #         focal_weights = focal_weights[None,:,None].expand(y_hat.shape[0], -1, y_hat.shape[-1])
    #         squared_errors = ((y_hat-true_pred)**2.0)
            
    #         squared_errors = torch.where(mask, squared_errors, torch.nan)
    #         focal_weights = torch.where(mask, focal_weights, torch.nan)
            
    #         lead_time_mse = torch.nansum(squared_errors * focal_weights, dim = 1)/torch.nansum(focal_weights, dim = 1)
    #         total_mse = torch.nanmean(lead_time_mse)
    #         return total_mse
    #     else:
    #         return torch.sum((y_hat[mask]-true_pred[mask])**2.0)  / torch.sum(mask)


def masked_mse(y_hat, y, mask):
    # y_hat = y_hat.to(device)
    # y = y.to(device)
    # mask = mask.to(device)
    return torch.sum((y_hat[mask]-y[mask])**2.0)  / torch.sum(mask)


# Gradient computation with autograd - more efficient
def jacobian_batches(output, input, create_graph = True):
        
    grad_outputs = torch.eye(output.shape[-1])[:,None,:].expand(-1, output.shape[0], -1).to(output.device)
    return torch.autograd.grad(output, input, grad_outputs = grad_outputs,
                               create_graph  = create_graph,
                               retain_graph = True,
                               is_grads_batched=True)

def physics_loss(y_hat, coord_input, g = torch.tensor([0]),
                 k_lat = torch.tensor([1]),
                 k_lon = torch.tensor([1]),
                 S_y = torch.tensor([1]),
                 create_graph_lastj = True):
    # compute gradients
    jacobian_lat_lon_dtm = jacobian_batches(output = y_hat,
                 input = coord_input)[0]
    
    h_dlat = torch.moveaxis(jacobian_lat_lon_dtm[:,:,0], 0, 1)
    h_dlat = -k_lat * h_dlat #* y_hat
    
    h_dlon = torch.moveaxis(jacobian_lat_lon_dtm[:,:,1], 0, 1)
    h_dlon = -k_lon * h_dlon #* y_hat
    
    
    h_d2lat = jacobian_batches(output = h_dlat,
                 input = coord_input,
                 create_graph = create_graph_lastj)[0][:,:,0]
    h_d2lat = torch.moveaxis(h_d2lat, 0, 1)
    
    h_d2lon = jacobian_batches(output = h_dlon,
                 input = coord_input,
                 create_graph = create_graph_lastj)[0][:,:,1]
    h_d2lon = torch.moveaxis(h_d2lon, 0, 1)
    
    
    h_dt = y_hat[:,1:] - y_hat[:,:-1] # first diff
    
    pde_res = (S_y * h_dt) + h_d2lat[:,:-1] + h_d2lon[:,:-1] - g
    loss_pde = torch.sum(pde_res**2)
    
    return loss_pde

def fdiff_fprime_soa(f_ahead, f_behind, delta):
    fprime = (f_ahead - f_behind)/(2*delta)
    return fprime

def fdiff_fsecond_soa(f, f_ahead, f_behind, delta):
    fsecond = (f_ahead + f_behind -2*f)/(delta**2)
    return fsecond

def disc_physics_loss(y_hat,
                      y_hat_two_right, y_hat_two_left, y_hat_two_up, y_hat_two_down,
                      k_lat_up, k_lat_down,
                      k_lon_right, k_lon_left,
                      step = torch.tensor([1]),
                      g =  torch.tensor([0]), S_y =  torch.tensor([1])):
    
    
    #first_lon_diff = - k_lon * fdiff_fprime_soa(y_hat_right, y_hat_left, delta = step)
    first_lon_diff_right = - k_lon_right * fdiff_fprime_soa(y_hat_two_right, y_hat, delta = step) # negative for inc wr
    first_lon_diff_left = - k_lon_left * fdiff_fprime_soa(y_hat, y_hat_two_left, delta = step) # negative for dec wr
    
    #first_lat_diff = - k_lat * fdiff_fprime_soa(y_hat_up, y_hat_down, delta = step)
    first_lat_diff_up = - k_lat_up * fdiff_fprime_soa(y_hat_two_up, y_hat, delta = step) # negative for inc wr
    first_lat_diff_down = - k_lat_down * fdiff_fprime_soa(y_hat, y_hat_two_down, delta = step) # negative for dec wr

    second_lon_diff = fdiff_fprime_soa(first_lon_diff_right, first_lon_diff_left, delta = step) # negative for inc wr
    second_lat_diff = fdiff_fprime_soa(first_lat_diff_up, first_lat_diff_down, delta = step)

    first_time_diff = S_y * (y_hat[:,:,1:] - y_hat[:,:,:-1]) # positive for inc wr
    
    residuals = first_time_diff + second_lon_diff[:,:,:-1] + second_lat_diff[:,:,:-1] + g
    loss_physics = torch.mean(residuals**2)
    return loss_physics

def fdif_conv(f, filter):
    
    filter = filter[None, None, :, :] # f(minibatch,in_channels,iH,iW)
    output = F.conv2d(f, filter)
    
    return output


if __name__ == "__main__":
    pass