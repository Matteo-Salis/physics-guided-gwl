import torch
import torch.nn as nn
from scipy import signal


def ConvLon(x, conv_type = "simple_gradient"):
    # horizontal contribution
    conv = torch.Tensor([[-1,2,-1],[-1,2,-1],[-1,2,-1]] )
    if conv_type == "centered_contribution":
        conv = torch.Tensor([[-1,0,-1],[-1,6,-1],[-1,0,-1]] )
    if conv_type == "simple_gradient":
        conv = torch.Tensor([[0,0,0],[0,-1,1],[0,0,0]] )

    out = torch.Tensor(x.shape)

    for b in range(x.shape[0]):
        for t in range(x.shape[2]):
            out[b,0,t,:,:] = torch.from_numpy(signal.convolve2d(x[b,0,t,:,:].cpu().detach().numpy(), conv, mode='same'))
    return out

def ConvLat(x, conv_type = "simple_gradient"):
    # vertical contribution
    conv = torch.Tensor([[-1,-1,-1],[2,2,2],[-1,-1,-1]] )
    if conv_type == "centered_contribution":
        conv = torch.Tensor([[-1,-1,-1],[0,6,0],[-1,-1,-1]] )
    if conv_type == "simple_gradient":
        conv = torch.Tensor([[0,1,0],[0,-1,0],[0,0,0]] )

    out = torch.Tensor(x.shape)

    for b in range(x.shape[0]):
        for t in range(x.shape[2]):
            out[b,0,t,:,:] = torch.from_numpy(signal.convolve2d(x[b,0,t,:,:].cpu().detach().numpy(), conv, mode='same'))

    return out

def pde_grad_loss(y_hat, device = "cuda"):
    # already normalized and computed on height
    y_hat.to(device)

    lat_grad = ConvLat(y_hat).to(device)
    lon_grad = ConvLon(y_hat).to(device)

    # y_hat_t_1 - y_hat_t = lat_grad_t + lon_grad_t
    loss = torch.sum(y_hat[:,0,1:-1,:,:] - y_hat[:,0,0:-2,:,:] + lat_grad[:,0,0:-2,:,:] + lon_grad[:,0,0:-2,:,:] )

    return torch.abs(loss)

def pde_grad_loss_darcy(y_hat, device = "cuda"):
    # normalized and working on height
    # print(y_hat.shape)

    y_hat.to(device)

    k_x = 1  # TODO: to esimate
    k_y = 1  # TODO: to esimate

    lat_grad = ConvLat(y_hat).to(device)
    lon_grad = ConvLon(y_hat).to(device)

    dh_t = y_hat[:,0,1:-1,:,:] - y_hat[:,0,0:-2,:,:]
    dh_x = lon_grad[:,0,0:-2,:,:]
    d2_x = ConvLon(torch.unsqueeze(dh_x, dim=1)).to(device)
    dh_y = lat_grad[:,0,0:-2,:,:]
    d2_y = ConvLat(torch.unsqueeze(dh_y, dim=1)).to(device)

    G = 0 # TODO: to esimate

    loss = dh_t - (d2_x * (-k_x * dh_x * y_hat[:,0,0:-2,:,:]) ) - (d2_y * (-k_y * dh_y * y_hat[:,0,0:-2,:,:]))
    loss = torch.sum(loss ** 2) / torch.numel(loss)

    return loss

def pde_grad_loss_wtd(y_hat, dtm, wtd_mean, wtd_std, device = "cuda"):
    # TODO: normalized or de-normalized?
    # print(y_hat.shape)

    predict = (y_hat * wtd_std) + wtd_mean # denormalized
    predict[:,0,:,:,:] = - predict[:,0,:,:,:] + dtm[0,:,:]
    predict.to(device)

    lat_grad = ConvLat(predict).to(device)
    lon_grad = ConvLon(predict).to(device)

    # y_hat_t_1 - y_hat_t = lat_grad_t + lon_grad_t
    loss = torch.sum(predict[:,0,1:-1,:,:] - predict[:,0,0:-2,:,:] + lat_grad[:,0,0:-2,:,:] + lon_grad[:,0,0:-2,:,:] )

    return torch.abs(loss)

def loss_positive_height(y, mean, std, device = "cuda"):
    predict = torch.unsqueeze(y[:,0,:,:,:], dim=1).to(device)
    predict = (predict * torch.from_numpy(std)) + torch.from_numpy(mean)
    errors = torch.clamp(predict, max=0)

    loss = torch.sum( errors ** 2.0 ) / torch.numel(errors)
    return loss

def loss_super_res(y_super_res, weather, device = "cuda"):
    f = nn.AdaptiveMaxPool3d((None, weather.shape[-2], weather.shape[-1]))
    errors = f(y_super_res) - weather
    return torch.sum( errors ** 2.0 ) / torch.numel(errors)

def loss_masked(y_hat, y, device = "cuda"):
    predict = torch.unsqueeze(y[:,0,:,:,:], dim=1).to(device)
    target = y_hat.to(device)
    mask = y[:,1,:,:,:].bool().to(device)
    out = (torch.sum( (predict - target) * mask) ** 2.0 ) / torch.sum(mask)
    return out