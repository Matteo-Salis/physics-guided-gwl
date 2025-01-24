import numpy as np
import torch


def masked_mse(y_hat, y, mask):
    # y_hat = y_hat.to(device)
    # y = y.to(device)
    # mask = mask.to(device)
    return torch.sum(((y_hat[mask]-y[mask]))**2.0)  / torch.sum(mask)


# Gradient computation with autograd - more efficient
def jacobian_batches(output, input):
    
    grad_outputs = torch.eye(output.shape[-1])[:,None,:].expand(-1, output.shape[0], -1).to(output.device)
    return torch.autograd.grad(output, input, grad_outputs = grad_outputs,
                               create_graph  = False,
                               retain_graph = True,
                               is_grads_batched=True)

def physics_loss(y_hat, coord_input, g = torch.tensor([0]),
                 k_lat = torch.tensor([1]),
                 k_lon = torch.tensor([1]),
                 S_y = torch.tensor([1])):
    # compute gradients
    jacobian_lat_lon_dtm = jacobian_batches(output = y_hat,
                 input = coord_input)[0]
    #print("firs J computed")
    h_dlat = torch.moveaxis(jacobian_lat_lon_dtm[:,:,0], 0, 1)
    h_dlat = -(h_dlat * y_hat) * k_lat
    #print("h_dlat", h_dlat[:,:3])
    #print("h_dlat", h_dlat.shape)
    h_dlon = torch.moveaxis(jacobian_lat_lon_dtm[:,:,1], 0, 1)
    h_dlon = -(h_dlon * y_hat) * k_lon
    #print("h_dlon", h_dlon[:,:3])
    
    h_d2lat = jacobian_batches(output = h_dlat,
                 input = coord_input)[0][:,:,0]
    h_d2lat = torch.moveaxis(h_d2lat, 0, 1)
    #print("Second J-y computed")
    h_d2lon = jacobian_batches(output = h_dlon,
                 input = coord_input)[0][:,:,1]
    h_d2lon = torch.moveaxis(h_d2lon, 0, 1)
    #print("Second J-x computed")
    h_dt = y_hat[:,1:] - y_hat[:,:-1] # first diff
    
    pde_res = (S_y * h_dt) + h_d2lat[:,:-1] + h_d2lon[:,:-1] - g
    loss_pde = torch.sum(pde_res**2)
    
    return loss_pde
