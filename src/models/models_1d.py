import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.dataset_1d import Dataset_1D

class SC_LSTM_idw(nn.Module):
    
    def __init__(self,
                 timestep = 180,
                 cb_fc_layer = 5,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 lstm_layer = 5,
                 lstm_input_units = 16,
                 lstm_units = 32,
                 ):
        super().__init__()
        
        self.timestep = timestep
        self.lstm_layer = lstm_layer
        self.lstm_input_units = lstm_input_units
        self.lstm_units = lstm_units
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.Tanh())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(16, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.LeakyReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.LeakyReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, int(self.conv_filters), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters), int(self.conv_filters/2), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters/2)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters/2), self.lstm_input_units, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.lstm_input_units))
        conv3d_stack.append(nn.LeakyReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)
        
    def idw(self, dist, values, x_mask, weight_std = True):
        
        weights = 1/(dist + torch.tensor([1e-8]).to(torch.float32).to(dist.device))
        weights = weights * x_mask
        numerator = torch.sum(weights*values, dim = 1)
        denominator = torch.sum(weights, dim = 1)
        output = numerator/denominator
        weights_cv = torch.std(weights, dim = (1,2)) / torch.mean(weights, dim = (1,2))
            
        if weight_std is True:
                output = [output, weights_cv]
            
        return output


    def forward(self, x, z, w, x_mask):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        # Batch dimension
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        if len(z.shape) < 2:
            z = z.unsqueeze(0)
        
        if len(w[0].shape) < 5 and len(w[1].shape) < 4:
            w[0] = w[0].unsqueeze(0)
            w[1] = w[1].unsqueeze(0)
        
        if len(x_mask.shape) < 2:
            x_mask = x_mask.unsqueeze(0)
            
        # Conditioning block
        target_dist = torch.cdist(x[:,:,:3], z.unsqueeze(1), p=2.0) # (B×P×M), (B×R×M), OUTPUT: (B×P×R) 
        target0 = self.idw(dist = target_dist,
                          values = x[:,:,-1].unsqueeze(-1),
                          x_mask = x_mask.unsqueeze(-1),
                          weight_std = True)
        
        target0 = torch.cat([z,
                             target0[0],
                             target0[1].unsqueeze(1)], dim = -1)
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        wb_td3dconv = wb_td3dconv.squeeze((3,4))
        wb_td3dconv = torch.moveaxis(wb_td3dconv, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = torch.movedim(target0_h, 0, 1)
        
        target_ts = self.lstm_1(wb_td3dconv,
                                 (target0_h.contiguous(),
                                  torch.zeros_like(target0_h).to(target0_h.device))) #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        target_ts_out = self.fc(target_ts[0])
        
        return target_ts_out.squeeze()
    

############## MODEL 2 ##############

class SC_LSTM_att(nn.Module):
    def __init__(self,
                 timestep = 180,
                 cb_emb_dim = 16,
                 cb_att_h = 4,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 lstm_layer = 1,
                 lstm_input_units = 16,
                 lstm_units = 128,
                 ):
        super().__init__()
        
        self.timestep = timestep
        self.lstm_layer = lstm_layer
        self.lstm_input_units = lstm_input_units
        self.lstm_units = lstm_units
        self.cb_emb_dim = cb_emb_dim
        self.cb_att_h = cb_att_h
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        
        # Attention block
        if self.cb_emb_dim is not None:
            cb_emb = []
            cb_emb.append(nn.Linear(3, self.cb_emb_dim))
            cb_emb.append(nn.LeakyReLU())
            self.cb_emb = nn.Sequential(*cb_emb)
            edim = self.cb_emb_dim
            
        else:
            edim = 3
        
        self.multihead_att = nn.MultiheadAttention(edim, self.cb_att_h,
                                                   batch_first=True, vdim=1)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(edim + 4, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
            
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.Tanh())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(16, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.LeakyReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.LeakyReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, int(self.conv_filters), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters), int(self.conv_filters/2), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters/2)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters/2), self.lstm_input_units, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.lstm_input_units))
        conv3d_stack.append(nn.LeakyReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)


    def forward(self, x, z, w, x_mask):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        # Batch dimension
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        if len(z.shape) < 2:
            z = z.unsqueeze(0)
        
        if len(w[0].shape) < 5 and len(w[1].shape) < 4:
            w[0] = w[0].unsqueeze(0)
            w[1] = w[1].unsqueeze(0)
        
        if len(x_mask.shape) < 2:
            x_mask = x_mask.unsqueeze(0)
            
        # Conditioning block
        
        if self.cb_emb_dim is not None:
            coords = torch.cat([x[:,:,:3],
                                z.unsqueeze(1)], dim = 1)
            
            cb_emb = self.cb_emb(coords)
            query = cb_emb[:,coords.shape[1]-1,:]
            keys = cb_emb[:,:coords.shape[1]-1,:]
            
        else:
            query = z
            keys = x[:,:,:3]
        
        attn_output, attn_output_weights = self.multihead_att(query = query.unsqueeze(1), #(N,L,E)
                                                  key = keys,
                                                  value = x[:,:,-1].unsqueeze(-1),
                                                  key_padding_mask = ~x_mask, #(N,S)
                                                   )
        
        weights_cv = torch.std(attn_output_weights, dim = (1,2)) / torch.mean(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_cv.unsqueeze(1)], dim = -1)
        
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        wb_td3dconv = wb_td3dconv.squeeze((3,4))
        wb_td3dconv = torch.moveaxis(wb_td3dconv, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = torch.movedim(target0_h, 0, 1)
        
        target_ts = self.lstm_1(wb_td3dconv,
                                 (target0_h.contiguous(),
                                  torch.zeros_like(target0_h).to(target0_h.device)))  #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        target_ts_out = self.fc(target_ts[0])
        
        return target_ts_out.squeeze()
    

############## MODEL 3 ############## 

class CausalConv1d(torch.nn.Conv1d):
    # inspired by https://github.com/pytorch/pytorch/issues/1333
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.causal_padding = (kernel_size - 1) * dilation
        
    def causal_conditional_padding(self, conditioning, series, padding):
        
        conditioning = conditioning.unsqueeze(-1).expand(-1, -1, padding)
        conditioned_padded_series = torch.cat([conditioning,
                                               series], dim = -1)
        
        return conditioned_padded_series
        
    def forward(self, input):

        padded_series = self.causal_conditional_padding(input[1],
                                                        input[0],
                                                        self.causal_padding)
        
        return super(CausalConv1d, self).forward(padded_series)

class SC_CCNN_att(nn.Module):
    def __init__(self,
                 timestep = 180,
                 cb_emb_dim = 16,
                 cb_att_h = 4,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 ccnn_input_filters = 8,
                 ccnn_kernel_size = 7,
                 ccnn_n_filters = 16,
                 ccnn_n_layers = 4,
                 ):
        super().__init__()
        
        self.timestep = timestep
        self.ccnn_input_filters = ccnn_input_filters
        self.ccnn_kernel_size = ccnn_kernel_size
        self.ccnn_n_filters = ccnn_n_filters
        self.ccnn_n_layers = ccnn_n_layers
        self.cb_emb_dim = cb_emb_dim
        self.cb_att_h = cb_att_h
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        
        # Attention block
        if self.cb_emb_dim is not None:
            cb_emb = []
            cb_emb.append(nn.Linear(3, self.cb_emb_dim))
            cb_emb.append(nn.LeakyReLU())
            self.cb_emb = nn.Sequential(*cb_emb)
            edim = self.cb_emb_dim
            
        else:
            edim = 3
        
        self.multihead_att = nn.MultiheadAttention(edim, self.cb_att_h,
                                                   batch_first=True, vdim=1)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(edim + 4, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.ccnn_input_filters))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(16, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.LeakyReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.LeakyReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, int(self.conv_filters), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters), int(self.conv_filters/2), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters/2)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters/2), self.ccnn_input_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.ccnn_input_filters))
        conv3d_stack.append(nn.LeakyReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        
        for cl in range(self.ccnn_n_layers):
            setattr(self, f"convolution_1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               self.ccnn_kernel_size))
            setattr(self, f"convolution_1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                              self.ccnn_input_filters,
                                              1,
                                              padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        fc = []
        fc.append(nn.Linear(self.ccnn_input_filters, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, z, w, x_mask):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        # Batch dimension
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        if len(z.shape) < 2:
            z = z.unsqueeze(0)
        
        if len(w[0].shape) < 5 and len(w[1].shape) < 4:
            w[0] = w[0].unsqueeze(0)
            w[1] = w[1].unsqueeze(0)
        
        if len(x_mask.shape) < 2:
            x_mask = x_mask.unsqueeze(0)
            
        # Conditioning block
        
        if self.cb_emb_dim is not None:
            coords = torch.cat([x[:,:,:3],
                                z.unsqueeze(1)], dim = 1)
            
            cb_emb = self.cb_emb(coords)
            query = cb_emb[:,coords.shape[1]-1,:]
            keys = cb_emb[:,:coords.shape[1]-1,:]
            
        else:
            query = z
            keys = x[:,:,:3]
        
        attn_output, attn_output_weights = self.multihead_att(query = query.unsqueeze(1), #(N,L,E)
                                                  key = keys,
                                                  value = x[:,:,-1].unsqueeze(-1),
                                                  key_padding_mask = ~x_mask, #(N,S)
                                                   )
        
        weights_cv = torch.std(attn_output_weights, dim = (1,2)) / torch.mean(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_cv.unsqueeze(1)], dim = -1)
        
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        target_ts = wb_td3dconv.squeeze((3,4))
        
        # Sequential block
        for cl in range(self.ccnn_n_layers):
            target_ts = getattr(self, f"convolution_1d_{cl}")([target_ts, target0])
            target_ts = getattr(self, f"convolution_1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"conv1x1_{cl}")(target_ts)
        
        target_ts = torch.moveaxis(target_ts, -1, 1)
        target_ts_out = self.fc(target_ts)
        
        return target_ts_out.squeeze()
    
############## MODEL 4 ############## 

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.causal_padding = (kernel_size - 1) * dilation
        
    def causal_conditional_padding(self, conditioning, series, padding):
        
        conditioning = conditioning.unsqueeze(-1).expand(-1, -1, padding)
        conditioned_padded_series = torch.cat([conditioning,
                                               series], dim = -1)
        
        return conditioned_padded_series
        
    def forward(self, input):

        padded_series = self.causal_conditional_padding(input[1],
                                                        input[0],
                                                        self.causal_padding)
        
        return super(CausalConv1d, self).forward(padded_series)

class SC_CCNN_idw(nn.Module):
    
    def __init__(self,
                 timestep = 180,
                 cb_emb_dim = 16,
                 cb_att_h = 4,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 ccnn_input_filters = 8,
                 ccnn_kernel_size = 7,
                 ccnn_n_filters = 16,
                 ccnn_n_layers = 4,
                 ):
        super().__init__()
        
        self.timestep = timestep
        self.ccnn_input_filters = ccnn_input_filters
        self.ccnn_kernel_size = ccnn_kernel_size
        self.ccnn_n_filters = ccnn_n_filters
        self.ccnn_n_layers = ccnn_n_layers
        self.cb_emb_dim = cb_emb_dim
        self.cb_att_h = cb_att_h
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.ccnn_input_filters))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(16, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.LeakyReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.LeakyReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, int(self.conv_filters), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters), int(self.conv_filters/2), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters/2)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters/2), self.ccnn_input_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.ccnn_input_filters))
        conv3d_stack.append(nn.LeakyReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        for cl in range(self.ccnn_n_layers):
            setattr(self, f"convolution_1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               self.ccnn_kernel_size))
            setattr(self, f"batch_norm_1d_{cl}",
                    nn.BatchNorm1d(int(self.ccnn_n_filters)))
            
            setattr(self, f"convolution_1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                              self.ccnn_input_filters,
                                              1,
                                              padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        fc = []
        fc.append(nn.Linear(self.ccnn_input_filters, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)

    def idw(self, dist, values, x_mask, weight_std = True):
        
        weights = 1/(dist + torch.tensor([1e-8]).to(torch.float32).to(dist.device))
        weights = weights * x_mask
        numerator = torch.sum(weights*values, dim = 1)
        denominator = torch.sum(weights, dim = 1)
        output = numerator/denominator
        weights_cv = torch.std(weights, dim = (1,2)) / torch.mean(weights, dim = (1,2))
            
        if weight_std is True:
                output = [output, weights_cv]
            
        return output
    
    def forward(self, x, z, w, x_mask):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        # Batch dimension
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        if len(z.shape) < 2:
            z = z.unsqueeze(0)
        
        if len(w[0].shape) < 5 and len(w[1].shape) < 4:
            w[0] = w[0].unsqueeze(0)
            w[1] = w[1].unsqueeze(0)
        
        if len(x_mask.shape) < 2:
            x_mask = x_mask.unsqueeze(0)
            
        # Conditioning block
        
        target_dist = torch.cdist(x[:,:,:3], z.unsqueeze(1), p=2.0) # (B×P×M), (B×R×M), OUTPUT: (B×P×R) 
        target0 = self.idw(dist = target_dist,
                          values = x[:,:,-1].unsqueeze(-1),
                          x_mask = x_mask.unsqueeze(-1),
                          weight_std = True)
        
        target0 = torch.cat([z,
                             target0[0],
                             target0[1].unsqueeze(1)], dim = -1)
        
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        target_ts = wb_td3dconv.squeeze((3,4))
        
        # Sequential block
        for cl in range(self.ccnn_n_layers):
            target_ts = getattr(self, f"convolution_1d_{cl}")([target_ts, target0])
            target_ts = getattr(self, f"batch_norm_1d_{cl}")(target_ts)
            target_ts = getattr(self, f"convolution_1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"conv1x1_{cl}")(target_ts)
        
        target_ts = torch.moveaxis(target_ts, -1, 1)
        target_ts_out = self.fc(target_ts)
        
        return target_ts_out.squeeze()

######### Model 5 PI ################ 

class SC_PICCNN_att(nn.Module):
    def __init__(self,
                 ph_params,
                 timestep = 180,
                 cb_emb_dim = 16,
                 cb_att_h = 4,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 ccnn_input_filters = 8,
                 ccnn_kernel_size = 7,
                 ccnn_n_filters = 16,
                 ccnn_n_layers = 4,
                 ph_params_neurons = 16,
                 ):
        super().__init__()
        
        self.timestep = timestep
        self.ccnn_input_filters = ccnn_input_filters
        self.ccnn_kernel_size = ccnn_kernel_size
        self.ccnn_n_filters = ccnn_n_filters
        self.ccnn_n_layers = ccnn_n_layers
        self.cb_emb_dim = cb_emb_dim
        self.cb_att_h = cb_att_h
        self.cb_fc_layer = cb_fc_layer
        self.cb_fc_neurons = cb_fc_neurons
        self.conv_filters = conv_filters
        self.ph_params = ph_params
        self.ph_params_neurons = ph_params_neurons
        
        # Physics block
        self.ph_params_fc = nn.Sequential(nn.Linear(3, self.ph_params_neurons),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.ph_params_neurons, 2))
            
            # self.hydro_cond_lon = torch.FloatTensor([self.ph_params["hyd_cond"][0]])
            # self.hydro_cond_lat = torch.FloatTensor([self.ph_params["hyd_cond"][0]])
            
        # Attention block
        if self.cb_emb_dim is not None:
            cb_emb = []
            cb_emb.append(nn.Linear(3, self.cb_emb_dim))
            cb_emb.append(nn.LeakyReLU())
            self.cb_emb = nn.Sequential(*cb_emb)
            edim = self.cb_emb_dim
            
        else:
            edim = 3
        
        self.multihead_att = nn.MultiheadAttention(edim, self.cb_att_h,
                                                   batch_first=True, vdim=1)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(edim + 4, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.ccnn_input_filters))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(16, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.LeakyReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.LeakyReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, int(self.conv_filters), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters), int(self.conv_filters/2), (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(int(self.conv_filters/2)))
        conv3d_stack.append(nn.LeakyReLU())
        conv3d_stack.append(nn.Conv3d(int(self.conv_filters/2), self.ccnn_input_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.ccnn_input_filters))
        conv3d_stack.append(nn.LeakyReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        
        for cl in range(self.ccnn_n_layers):
            setattr(self, f"convolution_1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               self.ccnn_kernel_size))
            setattr(self, f"batch_norm_1d_{cl}",
                    nn.BatchNorm1d(int(self.ccnn_n_filters)))
            
            setattr(self, f"convolution_1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                              self.ccnn_input_filters,
                                              1,
                                              padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        fc = []
        fc.append(nn.Linear(self.ccnn_input_filters, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, z, w, x_mask, hc_out = False):
        """
        input : x (31, 5); z (1, 3); w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        # Batch dimension
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        
        if len(z.shape) < 2:
            z = z.unsqueeze(0)
        
        if len(w[0].shape) < 5 and len(w[1].shape) < 4:
            w[0] = w[0].unsqueeze(0)
            w[1] = w[1].unsqueeze(0)
        
        if len(x_mask.shape) < 2:
            x_mask = x_mask.unsqueeze(0)
            
            
        # Conditioning block
        
        if self.cb_emb_dim is not None:
            coords = torch.cat([x[:,:,:3],
                                z.unsqueeze(1)], dim = 1)
            
            cb_emb = self.cb_emb(coords)
            query = cb_emb[:,coords.shape[1]-1,:]
            keys = cb_emb[:,:coords.shape[1]-1,:]
            
        else:
            query = z
            keys = x[:,:,:3]
        
        attn_output, attn_output_weights = self.multihead_att(query = query.unsqueeze(1), #(N,L,E)
                                                  key = keys,
                                                  value = x[:,:,-1].unsqueeze(-1),
                                                  key_padding_mask = ~x_mask, #(N,S)
                                                   )
        
        weights_cv = torch.std(attn_output_weights, dim = (1,2)) / torch.mean(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_cv.unsqueeze(1)], dim = -1)
        
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        target_ts = wb_td3dconv.squeeze((3,4))
        
        # Sequential block
        for cl in range(self.ccnn_n_layers):
            target_ts = getattr(self, f"convolution_1d_{cl}")([target_ts, target0])
            target_ts = getattr(self, f"batch_norm_1d_{cl}")(target_ts)
            target_ts = getattr(self, f"convolution_1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"conv1x1_{cl}")(target_ts)
        
        target_ts = torch.moveaxis(target_ts, -1, 1)
        target_ts_out = self.fc(target_ts).squeeze()

        
        # Physics Block
        hyd_cond = self.ph_params_fc(z)
        hyd_cond = torch.clamp(hyd_cond,
                                min=self.ph_params["hyd_cond"][1],
                                max=self.ph_params["hyd_cond"][2])
        
        hyd_cond = hyd_cond + torch.ones_like(hyd_cond)*self.ph_params["hyd_cond"][0]
            
        if hc_out is True:    
            return target_ts_out, hyd_cond
        
        else: 
            return target_ts_out
        

###########################################
    
if __name__ == "__main__":
    print("Loading data.json...")
    config = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        config = json.load(f)
    print(f"Read data.json: {config}")

    print("Loading ContinuousDataset...")
    ds = Dataset_1D(config)
    x, z, w_values, y, x_mask, y_mask  = ds[0]
    
    weather_coords = ds.get_weather_coords()
    weather_dtm = ds.get_weather_dtm()
    weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)
    weather_coords_batch = weather_coords.unsqueeze(0)
    w = [w_values, weather_coords_batch]
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading Continuous1DNN...")
    timesteps = config["timesteps"]
    model = SC_LSTM_idw(timestep = config["timesteps"],
                 cb_fc_layer = config["cb_fc_layer"], #5,
                 cb_fc_neurons = config["cb_fc_neurons"], # 32,
                 conv_filters = config["conv_filters"], #32,
                 lstm_layer = config["lstm_layer"], #5,
                 lstm_input_units = config["lstm_input_units"], #16,
                 lstm_units = config["lstm_units"] #32
                 ).to(device)
    print("Continuous1DNN prediction...")
    y = model(x, z, w, x_mask)
    print(f"Output:\n{y}")
    