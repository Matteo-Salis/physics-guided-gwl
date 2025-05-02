import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.dataset_1d import Dataset_1D

##### Blocks ######

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

## Weather block
    
class WeatherBlock(nn.Module):
    def __init__(self, conv_filters, output_filters):
        super(WeatherBlock, self).__init__()

        
        self.conv3d_1 = nn.Conv3d(16, conv_filters, (1,2,2)) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        self.conv3d_bn_1 = nn.LayerNorm(int(conv_filters))
        self.conv3d_a_1 = nn.LeakyReLU()
        
        for i in range(4):
            setattr(self, f"conv3d_{i+2}",
                    nn.Conv3d(conv_filters, conv_filters, (1,2,2)))
            setattr(self, f"conv3d_bn_{i+2}",
                    nn.LayerNorm(int(conv_filters)))
            setattr(self, f"conv3d_a_{i+2}",
                    nn.LeakyReLU())
        
        self.conv3d_ap = nn.AdaptiveAvgPool3d((None,4,4))
        self.conv3d_skip_1 = nn.AdaptiveAvgPool3d((None,4,4))    
        
        self.conv3d_6 = nn.Conv3d(conv_filters, conv_filters, (1,1,1))
        self.conv3d_6_bn = nn.LayerNorm(int(conv_filters)) #nn.BatchNorm3d(int(conv_filters))
        self.conv3d_6_a = nn.LeakyReLU()
        
        self.conv3d_skip_2 = nn.AdaptiveAvgPool3d((None,1,1))
        
        self.conv3d_7 = nn.Conv3d(conv_filters, conv_filters, (1,2,2))
        self.conv3d_7_bn = nn.LayerNorm(int(conv_filters))
        self.conv3d_7_a = nn.LeakyReLU()
        
        self.conv3d_8 = nn.Conv3d(conv_filters, int(conv_filters), (1,2,2))
        self.conv3d_8_bn = nn.LayerNorm(int(conv_filters))
        self.conv3d_8_a = nn.LeakyReLU()
        
        self.conv3d_9 = nn.Conv3d(int(conv_filters), output_filters, (1,2,2))
        self.conv3d_9_bn = nn.LayerNorm(int(output_filters))
        self.conv3d_9_a = nn.LeakyReLU()
  
    def forward(self, x):
        
        conv3d_out = self.conv3d_1(x)
        conv3d_out = self.conv3d_bn_1(torch.moveaxis(conv3d_out, 1, -1))
        conv3d_out = self.conv3d_a_1(torch.moveaxis(conv3d_out, -1, 1))
        
        conv_3d_skip_1 = self.conv3d_skip_1(torch.clone(conv3d_out))
        
        for i in range(4):
            conv3d_out = getattr(self, f"conv3d_{i+2}")(conv3d_out)
            conv3d_out = getattr(self, f"conv3d_bn_{i+2}")(torch.moveaxis(conv3d_out, 1, -1))
            conv3d_out = getattr(self, f"conv3d_a_{i+2}")(torch.moveaxis(conv3d_out, -1, 1))
        
        conv3d_out = self.conv3d_ap(conv_3d_skip_1)
        #skip 1
        conv3d_out = conv3d_out + conv_3d_skip_1
        
        conv3d_out = self.conv3d_6(conv3d_out)
        conv3d_out = self.conv3d_6_bn(torch.moveaxis(conv3d_out, 1, -1))
        conv3d_out = self.conv3d_6_a(torch.moveaxis(conv3d_out, -1, 1))
        
        conv_3d_skip_2 = self.conv3d_skip_2(torch.clone(conv3d_out))
        
        conv3d_out = self.conv3d_7(conv3d_out)
        conv3d_out = self.conv3d_7_bn(torch.moveaxis(conv3d_out, 1, -1))
        conv3d_out = self.conv3d_7_a(torch.moveaxis(conv3d_out, -1, 1))
        
        conv3d_out = self.conv3d_8(conv3d_out)
        conv3d_out = self.conv3d_8_bn(torch.moveaxis(conv3d_out, 1, -1))
        conv3d_out = self.conv3d_8_a(torch.moveaxis(conv3d_out, -1, 1))
        
        # skip 2
        conv3d_out = conv3d_out + conv_3d_skip_2
        
        conv3d_out = self.conv3d_9(conv3d_out)
        conv3d_out = self.conv3d_9_bn(torch.moveaxis(conv3d_out, 1, -1))
        conv3d_out = self.conv3d_9_a(torch.moveaxis(conv3d_out, -1, 1))
        
        return conv3d_out
    
############################################ MODEL 1 ########################################################

class SC_LSTM_idw(nn.Module):
    
    def __init__(self,
                 timestep = 180,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 lstm_layer = 1,
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
        cb_fc.append(nn.Linear(6, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.Tanh())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.lstm_input_units)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.LeakyReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)
        
    def idw(self, dist, values, x_mask, weight_stats = True):
        
        weights = 1/(dist + torch.tensor([1e-8]).to(torch.float32).to(dist.device))
        weights = weights * x_mask
        numerator = torch.sum(weights*values, dim = 1)
        denominator = torch.sum(weights, dim = 1)
        output = numerator/denominator
        weights_mean = torch.mean(weights, dim = (1,2))
        weights_std = torch.std(weights, dim = (1,2))
            
        if weight_stats is True:
                output = [output, weights_mean, weights_std]
            
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
        target0, weights_mean, weights_std  = self.idw(dist = target_dist,
                          values = x[:,:,-1].unsqueeze(-1),
                          x_mask = x_mask.unsqueeze(-1),
                          weight_stats = True)
        
        target0 = torch.cat([z,
                             target0,
                             weights_mean.unsqueeze(1),
                             weights_std.unsqueeze(1)], dim = -1)
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        weather_block_out = weather_block_out.squeeze((3,4))
        weather_block_out = torch.moveaxis(weather_block_out, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = torch.movedim(target0_h, 0, 1)
        
        target_ts = self.lstm_1(weather_block_out,
                                 (target0_h.contiguous(),
                                  torch.zeros_like(target0_h).to(target0_h.device))) #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        target_ts_out = self.fc(target_ts[0])
        
        return target_ts_out.squeeze()
    

############################################ MODEL 2 ########################################################

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
                 lstm_units = 32,
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
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape= edim + 5)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(edim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
            
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape= self.lstm_units)
        
        # Sequental Conditioning Block
        # self.cb_lstm = nn.LSTM(self.lstm_input_units, self.lstm_units,
        #                       batch_first=True,
        #                       num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        # Sequential Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.lstm_input_units)
            
        # Sequental conditioning block
        self.weather_lstm = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        # Sequential joint block
        # self.joint_lstm = nn.LSTM(self.lstm_units, self.lstm_units*2,
        #                                batch_first=True, 
        #                                num_layers=self.lstm_layer)
        
        if self.lstm_units > 8:
            fc = []
            fc.append(nn.Linear(self.lstm_units, 8))
            fc.append(nn.LeakyReLU())
            fc.append(nn.Linear(8, 1))
            self.fc = nn.Sequential(*fc)
        else:
            self.fc = nn.Linear(self.lstm_units, 1)


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
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        
        target0 = self.layer_norm_1(target0)
        
        target0 = self.cb_fc(target0)
        
        target0 = self.layer_norm_2(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        weather_block_out = weather_block_out.squeeze((3,4))
        weather_block_out = torch.moveaxis(weather_block_out, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = torch.movedim(target0_h, 0, 1)
        
        weather_block_ts = self.weather_lstm(weather_block_out,  #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
                                (target0_h.contiguous(),    
                                 torch.zeros_like(target0_h).to(target0_h.device))) #torch.zeros_like(target0_h).to(target0_h.device)
        
        # Sequential conditioning block
        # target_ts = self.cb_lstm(torch.zeros_like(weather_block_out),
        #                          (target0_h.contiguous(),
        #                           target0_h.contiguous()))
        
        # Block join
        #target_ts_out = target_ts[0] + weather_block_ts[0]
        
        #target_ts_out = self.joint_lstm(target_ts_out)
        
        target_ts_out = self.fc(weather_block_ts[0])
        
        return target_ts_out.squeeze()
    
    
################################ MODEL 2 TEACHER TRAINING ####################################################

class SC_LSTM_att_TT(nn.Module):
    def __init__(self,
                 timestep = 180,
                 cb_emb_dim = 16,
                 cb_att_h = 4,
                 cb_fc_layer = 2,
                 cb_fc_neurons = 32,
                 conv_filters = 32,
                 lstm_layer = 1,
                 lstm_input_units = 16,
                 lstm_units = 32,
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
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape= edim + 5)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(edim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
            
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape= self.lstm_units)
        
        # Sequental Conditioning Block
        # self.cb_lstm = nn.LSTM(self.lstm_input_units, self.lstm_units,
        #                       batch_first=True,
        #                       num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        # Sequential Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = int(self.lstm_input_units/2))
            
        # Sequental conditioning block
        self.weather_lstm = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        # Sequential joint block
        # self.joint_lstm = nn.LSTM(self.lstm_units, self.lstm_units*2,
        #                                batch_first=True, 
        #                                num_layers=self.lstm_layer)
        
        if self.lstm_units > 8:
            fc = []
            fc.append(nn.Linear(self.lstm_units, 8))
            fc.append(nn.LeakyReLU())
            fc.append(nn.Linear(8, 1))
            self.fc = nn.Sequential(*fc)
        else:
            self.fc = nn.Linear(self.lstm_units, 1)


    def forward(self, x, z, w, x_mask, y = None):
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
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        
        target0 = self.layer_norm_1(target0)
        
        target0 = self.cb_fc(target0)
        
        target0 = self.layer_norm_2(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        weather_block_out = weather_block_out.squeeze((3,4))
        weather_block_out = torch.moveaxis(weather_block_out, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = torch.movedim(target0_h, 0, 1)
        
        # if y is not None: 
        #     lstm_input = torch.concat([weather_block_out,
        #                                y.unsqueeze(-1)], dim = -1 )
            
        #     weather_block_ts = self.weather_lstm(lstm_input,  #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        #                                         (target0_h.contiguous(),
        #                                         target0_h.contiguous()))
            
        #     lstm_outputs = self.fc(weather_block_ts[0])
            
        h_lstm_input = self.fc(target0.unsqueeze(1))
        lstm_outputs = []
        lstm_hidden = (target0_h.contiguous(),    
                       target0_h.contiguous()) # torch.zeros_like(target0_h).to(target0_h.device)
    
        for tstep in range(self.timestep):
            
            lstm_output, lstm_hidden = self.weather_lstm(
                                                torch.cat([weather_block_out[:,tstep,:].unsqueeze(1),
                                                            h_lstm_input], dim = -1),
                                                lstm_hidden)
            lstm_output = self.fc(lstm_output)            
            lstm_outputs.append(lstm_output)
            
            if y is not None: 
                h_lstm_input = y[0][:,tstep][:,None, None]
                
                h_lstm_input_mask = ~y[1][:,tstep][:,None, None]
                if (h_lstm_input_mask).any():
                    h_lstm_input[h_lstm_input_mask] = lstm_output.detach()[h_lstm_input_mask]
            else:
                h_lstm_input = lstm_output.detach()
                
                
        lstm_outputs = torch.cat(lstm_outputs, dim=1)
            
        
        return lstm_outputs.squeeze()
                
            
            # weather_block_ts = self.weather_lstm(weather_block_out,  #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
            #                         (target0_h.contiguous(),    
            #                         target0_h.contiguous())) #torch.zeros_like(target0_h).to(target0_h.device)
            
        # Sequential conditioning block
        # target_ts = self.cb_lstm(torch.zeros_like(weather_block_out),
        #                          (target0_h.contiguous(),
        #                           target0_h.contiguous()))
        
        # Block join
        #target_ts_out = target_ts[0] + weather_block_ts[0]
        
        #target_ts_out = self.joint_lstm(target_ts_out)
        
        # target_ts_out = self.fc(weather_block_ts[0])
        
        # return target_ts_out.squeeze()
    

############################################ MODEL 3 ########################################################
    
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
        cb_emb_coord = []
        cb_emb_coord.append(nn.Linear(3, self.cb_emb_dim))
        cb_emb_coord.append(nn.LeakyReLU())
        self.cb_emb_coord = nn.Sequential(*cb_emb_coord)
        
        cb_emb_value = []
        cb_emb_value.append(nn.Linear(1, self.cb_emb_dim))
        cb_emb_value.append(nn.LeakyReLU())
        self.cb_emb_coord = nn.Sequential(*cb_emb_value)
        
        self.multihead_att_1 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        # q, k, v tutti a 16, sommiamo value e output e concat coord (in 16 dim)
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        self.fc_att_1 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # q, k, v - q, k sono skip da primi solo coord e v sono valori in output da fc
        
        self.multihead_att_2 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        self.layer_norm_3 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        self.fc_att_2 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_4 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # qua q = 1, ~ decoder
        # k sono sempre skip coord originarie
        # v dal layer prima
        self.multihead_att_3 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        # concat come vecchio 
        
        self.layer_norm_5 = nn.LayerNorm(normalized_shape = self.cb_emb_dim + 5)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(self.cb_emb_dim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.ccnn_input_filters))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        self.layer_norm_6 = nn.LayerNorm(normalized_shape= self.ccnn_input_filters)
        
        # Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.ccnn_input_filters)
            
        # Joint sequential block
        
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
        
        self.fc = nn.Linear(self.ccnn_input_filters, 1)

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
        
        
        coords = torch.cat([x[:,:,:3],
                            z.unsqueeze(1)], dim = 1)
        
        cb_emb_coords = self.cb_emb_coord =(coords)
        
        query = cb_emb_coords[:,coords.shape[1]-1,:]
        keys = cb_emb_coords[:,:coords.shape[1]-1,:]
        
        cb_emb_values = self.cb_emb_value(x[:,:,-1])
        
        att_values_1, _ = self.multihead_att_1(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = cb_emb_values,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_1 = att_values_1 + cb_emb_values
        att_values_1 = torch.cat([att_values_1, keys], dim = -1)
        att_values_1 = self.layer_norm_1(att_values_1)
        att_values_1 = self.fc_att_1(att_values_1)
        att_values_1 = self.layer_norm_2(att_values_1)
        
        att_values_2, _ = self.multihead_att_2(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = att_values_1,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_2 = att_values_2 + att_values_1
        att_values_2 = torch.cat([att_values_2, keys], dim = -1)
        att_values_2 = self.layer_norm_3(att_values_2)
        att_values_2 = self.fc_att_2(att_values_2)
        att_values_2 = self.layer_norm_4(att_values_2)
        
        att_values, attn_output_weights = self.multihead_att_3(
                                            query = query.unsqueeze(1), #(N,L,E)
                                            key = keys,
                                            value = att_values_2,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             att_values.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        target0 = self.layer_norm_5(target0)
        
        target0 = self.cb_fc(target0)
        
        target0 = self.layer_norm_6(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        target_ts = weather_block_out.squeeze((3,4))
        
        # Sequential block
        for cl in range(self.ccnn_n_layers):
            target_ts = getattr(self, f"convolution_1d_{cl}")([target_ts, target0])
            target_ts = getattr(self, f"convolution_1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"conv1x1_{cl}")(target_ts)
        
        target_ts = torch.moveaxis(target_ts, -1, 1)
        
        target_ts_out = self.fc(target_ts)
        
        return target_ts_out.squeeze()

############################################ MODEL 0 ########################################################   

class Weather_Upsampling_Block(nn.Module):
    def __init__(self,
                 input_channles = 10,
                 hidden_channels = 32,
                 output_channels = 16,
                 output_dim = [104, 150]):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.ConvTranspose3d(input_channles, hidden_channels, (1,5,5), stride=(1,1,1), dtype=torch.float32))
        # N, C, D, H, W
        self.layers.append(nn.BatchNorm3d(hidden_channels))# [C, H, W]
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.ConvTranspose3d(hidden_channels, hidden_channels, (1,3,3), stride=(1,3,3), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(hidden_channels))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.ConvTranspose3d(hidden_channels, hidden_channels, (1,3,3), stride=(1,3,3), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(hidden_channels))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Conv3d(hidden_channels, hidden_channels, (1,3,3), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(hidden_channels))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.AdaptiveAvgPool3d((None,output_dim[0],output_dim[1]))) #padding='same'
        
        self.layers.append(nn.Conv3d(hidden_channels, output_channels, (1,5,5), padding='same', dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(output_channels))
        self.layers.append(nn.LeakyReLU())
        
        self.block = nn.Sequential(*self.layers)
    
    def forward(self, input):
        return self.block(input)
    
class Conditioning_Attention_Block(nn.Module):
    
    def densification_dropout(self, sample, p = 0.25):
        
        """
        Dropout training as in densification of Andrychowicz et al. (2023)
        """
        
        new_X, new_X_mask = sample
        dropout_mask = torch.rand(new_X_mask.shape, device=new_X.device) > p
        
        new_X[:,:,-1] = new_X[:,:,-1] * dropout_mask
        new_X_mask = torch.logical_and(new_X_mask, dropout_mask)
        
        return new_X, new_X_mask
    
    def __init__(self,
                 embedding_dim,
                 heads,
                 hidden_channels,
                 output_channels,
                 densification_dropout_p
                 ):
        super().__init__()
        
        self.densification_dropout_p = densification_dropout_p
        
        cb_topo_emb = []
        cb_topo_emb.append(nn.Linear(3, embedding_dim))
        cb_topo_emb.append(nn.LeakyReLU())
        self.cb_topo_emb = nn.Sequential(*cb_topo_emb)
        
        cb_value_emb = []
        cb_value_emb.append(nn.Linear(4, embedding_dim))
        cb_value_emb.append(nn.LeakyReLU())
        self.cb_value_emb = nn.Sequential(*cb_value_emb)
        
        self.cb_multihead_att = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.cb_layer_norm_1 = nn.LayerNorm(normalized_shape = embedding_dim)
        
        self.cb_fc = nn.Sequential(nn.Linear(embedding_dim, hidden_channels),
                                      nn.LeakyReLU())
        
        self.cb_layer_norm_2 = nn.LayerNorm(normalized_shape = hidden_channels)
        
        self.cb_conv = nn.Sequential(nn.Conv2d(hidden_channels,
                                               output_channels,
                                               5,
                                               padding='same'),
                                     nn.BatchNorm2d(output_channels),
                                     nn.LeakyReLU())
        
    def forward(self, X, Z, X_mask):
            
            if self.training is True:
                X, X_mask = self.densification_dropout([X, X_mask],
                                                       p = self.densification_dropout_p)
            
            coords = torch.cat([X[:,:,:3],
                                Z.flatten(start_dim = 1, end_dim = 2)],
                               dim = 1)
            
            topographical_embedding = self.cb_topo_emb(coords)
            
            keys = topographical_embedding[:,:X.shape[1],:]
            queries = topographical_embedding[:,X.shape[1]:,:]
            values = self.cb_value_emb(X)
            
            target_Icond, _ = self.cb_multihead_att(
                                            query = queries, #(N,L,E)
                                            key = keys,
                                            value = values,
                                            key_padding_mask = ~X_mask, #(N,S)
                                            )
            
            target_Icond = self.cb_layer_norm_1(target_Icond)
            target_Icond = self.cb_fc(target_Icond)
            target_Icond = self.cb_layer_norm_2(target_Icond)
            target_Icond = torch.moveaxis(target_Icond, -1, 1)
            
            target_Icond = torch.reshape(target_Icond, (*target_Icond.shape[:2],
                                         Z.shape[1], Z.shape[2]))
            
            target_Icond = self.cb_conv(target_Icond)

            return target_Icond
    
class CausalConv3d(torch.nn.Conv1d):
    
    # TODO
    
    # inspired by https://github.com/pytorch/pytorch/issues/1333
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv3d, self).__init__(
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

# Inspired by https://github.com/czifan/ConvLSTM.pytorch/blob/master/networks/ConvLSTM.py (to cite)
class ConvLSTMBlock(nn.Module):
    def __init__(self, 
                 input_channles,
                 hidden_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = self._make_layer(input_channles+hidden_channels, hidden_channels*4,
                                       kernel_size, padding, stride)
        
        self.leaky_rely = nn.LeakyReLU()
        

    def _make_layer(self, input_channles, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(input_channles, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs, h_0 = None, c_0 = None):
        '''

        :param inputs: (N, C, D, H, W)
        :param hidden_state: (h_0: (N, C_out, H, W), c_0: (N, C_out, H, W))
        :return:
        '''
        outputs = []
        N, C, D, H, W = inputs.shape
        
        if h_0 is None:
            h_0 = torch.zeros(N, self.hidden_channels, H, W).to(inputs.device)
        if c_0 is None:
            c_0 = torch.zeros(N, self.hidden_channels, H, W).to(inputs.device)
        
        for t in range(D):
            combined = torch.cat([inputs[:,:,t,:,:], # (N, C, H, W)
                                  h_0], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_channels, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * c_0) + (ingate * cellgate)
            hy = outgate * self.leaky_rely(cy)
            outputs.append(hy)
            h_0 = hy
            c_0 = cy

        return torch.stack(outputs, dim = 2) # (N, C, D, H, W)
    
class Date_Conditioning_Block(nn.Module):
    def __init__(self,
                 n_channels):
        super().__init__()
        
        self.n_channels = n_channels
        
        for i in range(n_channels):
            
            setattr(self, f"fc_{i}",
                    nn.Linear(2, 2))
            
            
    def forward(self, input):
        output = []
        for i in range(self.n_channels):
            output.append(getattr(self, f"fc_{i}")(input))
        
        output = torch.stack(output, dim = 1)
        
        return output
        
        
    
class AttCB_ConvLSTM(nn.Module):
    def __init__(self,
                 cb_emb_dim = 16,
                 cb_heads = 4,
                 channels_cb_wb = 32,
                 convlstm_input_units = 16,
                 convlstm_units = 32,
                 densification_dropout = 0.25,
                 upsampling_dim = [104, 150]):
        
        super().__init__()
        
        self.convlstm_input_units = convlstm_input_units
        self.convlstm_units = convlstm_units
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb_wb = channels_cb_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 hidden_channels = self.channels_cb_wb,
                 output_channels = self.convlstm_units,
                 densification_dropout_p = self.densification_dropout_p)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(self.convlstm_input_units)
        
        ### Weather module ### 
        
        self.Weather_Module = Weather_Upsampling_Block(input_channles = 10,
                 hidden_channels = self.channels_cb_wb,
                 output_channels = self.convlstm_input_units,
                 output_dim = self.upsampling_dim)
        
        ### Sequential module ###
        
        self.convLSTM_1 = ConvLSTMBlock(input_channles = self.convlstm_input_units,
                 hidden_channels = self.convlstm_units,
                 kernel_size=3)
        
        self.Date_Conditioning_Module_sm = Date_Conditioning_Block(self.convlstm_units)
        
        self.convLSTM_2 =ConvLSTMBlock(input_channles = self.convlstm_units,
                 hidden_channels = 1,
                 kernel_size=3)
        
    def forward(self, X, Z, W, X_mask):
        
        ### Conditioning modules ###
        
        target_Icond = self.Icondition_Module(X, Z, X_mask)
        
        ### Weather module ### 
        
        Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # N, C, D, 2
        date_conditioning_wm = date_conditioning_wm[:,:,:,None, None,:].expand(-1, -1, -1,
                                                                         Weaether_seq.shape[3],
                                                                         Weaether_seq.shape[4],
                                                                         -1)
        
        Weaether_seq = (Weaether_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
        
        ### Sequential module ### 
        
        Output = self.convLSTM_1(Weaether_seq,
                                 target_Icond,
                                 target_Icond)
        
        date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        date_conditioning_sm = date_conditioning_sm[:,:,:,None, None,:].expand(-1, -1, -1,
                                                                         Output.shape[3],
                                                                         Output.shape[4],
                                                                         -1)
        
        Output = (Output * date_conditioning_sm[:,:,:,:,:,0]) + date_conditioning_sm[:,:,:,:,:,1]
        
        Output = self.convLSTM_2(Output)
        
        return Output
        
        
        

############################################ MODEL 4 ########################################################

class SC_CCNN_att_TRSP(nn.Module):
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
        cb_emb_coord = []
        cb_emb_coord.append(nn.Linear(3, self.cb_emb_dim))
        cb_emb_coord.append(nn.LeakyReLU())
        self.cb_emb_coord = nn.Sequential(*cb_emb_coord)
        
        cb_emb_value = []
        cb_emb_value.append(nn.Linear(1, self.cb_emb_dim))
        cb_emb_value.append(nn.LeakyReLU())
        self.cb_emb_value = nn.Sequential(*cb_emb_value)
        
        self.multihead_att_1 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        # q, k, v tutti a 16, sommiamo value e output e concat coord (in 16 dim)
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = self.cb_emb_dim*2)
        
        self.fc_att_1 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # q, k, v - q, k sono skip da primi solo coord e v sono valori in output da fc
        
        self.multihead_att_2 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        self.layer_norm_3 = nn.LayerNorm(normalized_shape = self.cb_emb_dim*2)
        
        self.fc_att_2 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_4 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # qua q = 1, ~ decoder
        # k sono sempre skip coord originarie
        # v dal layer prima
        self.multihead_att_3 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        # concat come vecchio 
        
        self.layer_norm_5 = nn.LayerNorm(normalized_shape = self.cb_emb_dim + 5)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(self.cb_emb_dim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        self.layer_norm_6 = nn.LayerNorm(normalized_shape= self.cb_fc_neurons)
        
        # Sequental Conditioning Block
        
        self.cb_transp_conv = nn.Sequential(
                                torch.nn.ConvTranspose1d(self.cb_fc_neurons,
                                                        self.cb_fc_neurons,
                                                        16,
                                                        stride=1,
                                                        padding=0,
                                                        bias=True,
                                                        dilation=1,
                                                        ),
                                nn.LeakyReLU(),
                                torch.nn.ConvTranspose1d(self.cb_fc_neurons,
                                                        self.cb_fc_neurons,
                                                        16,
                                                        stride=16,
                                                        padding=0,
                                                        bias=True,
                                                        dilation=1,
                                                        ),
                                nn.LeakyReLU(),
                                nn.Conv1d(self.cb_fc_neurons,
                                        self.ccnn_input_filters,
                                        1,
                                        padding="valid"),
                                nn.AdaptiveAvgPool1d(self.timestep)
                                )
        
        self.cb_layer_norm = nn.LayerNorm(self.ccnn_input_filters)
        
        for cl in range(int(self.ccnn_n_layers/2)):
            setattr(self, f"cb_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               kernel_size= 3,
                                               dilation = 2**cl))
            
            setattr(self, f"cb_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"cb_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        self.avg_pool_seq_cb = nn.AvgPool1d(kernel_size = self.ccnn_kernel_size,
                                            padding= self.ccnn_kernel_size//2,
                                            stride = 1,
                                            count_include_pad = False)
            
        self.layer_norm_seq_cb = nn.LayerNorm(self.ccnn_input_filters)
        
        # Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.ccnn_input_filters)
            
        
        for cl in range(int(self.ccnn_n_layers/2)):
            setattr(self, f"weather_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               3))
            
            setattr(self, f"weather_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"weather_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        self.layer_norm_seq_weather = nn.LayerNorm(self.ccnn_input_filters)    
        # Joint sequential block
        
        self.joint_1x1conv = nn.Conv1d(self.ccnn_input_filters*2,
                                        self.ccnn_input_filters,
                                        kernel_size= 1,
                                        padding="valid")
        
        self.joint_layer_norm_seq = nn.LayerNorm(int(ccnn_input_filters))
        self.joint_lrelu_seq = nn.LeakyReLU()
        
        
        for cl in range(self.ccnn_n_layers):
            setattr(self, f"joint_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               self.ccnn_kernel_size,
                                               dilation = 2**cl))
            
            setattr(self, f"joint_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"joint_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        self.fc = nn.Linear(self.ccnn_input_filters, 1)

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
        coords = torch.cat([x[:,:,:3],
                            z.unsqueeze(1)], dim = 1)
        
        cb_emb_coords = self.cb_emb_coord(coords)
        
        query = cb_emb_coords[:,coords.shape[1]-1,:]
        keys = cb_emb_coords[:,:coords.shape[1]-1,:]
        
        cb_emb_values = self.cb_emb_value(x[:,:,-1].unsqueeze(-1))
        
        att_values_1, _ = self.multihead_att_1(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = cb_emb_values,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_1 = att_values_1 + cb_emb_values
        att_values_1 = torch.cat([att_values_1, keys], dim = -1)
        att_values_1 = self.layer_norm_1(att_values_1)
        att_values_1 = self.fc_att_1(att_values_1)
        att_values_1 = self.layer_norm_2(att_values_1)
        
        att_values_2, _ = self.multihead_att_2(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = att_values_1,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_2 = att_values_2 + att_values_1
        att_values_2 = torch.cat([att_values_2, keys], dim = -1)
        att_values_2 = self.layer_norm_3(att_values_2)
        att_values_2 = self.fc_att_2(att_values_2)
        att_values_2 = self.layer_norm_4(att_values_2)
        
        att_values, attn_output_weights = self.multihead_att_3(
                                            query = query.unsqueeze(1), #(N,L,E)
                                            key = keys,
                                            value = att_values_2,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             att_values.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        target0 = self.layer_norm_5(target0)
        
        target0 = self.cb_fc(target0)
        
        target0 = self.layer_norm_6(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        # Sequential block
        weather_block_ts = weather_block_out.squeeze((3,4))
    
        for cl in range(int(self.ccnn_n_layers/2)):
            weather_block_ts = getattr(self, f"weather_conv1d_{cl}")([weather_block_ts, weather_block_ts[:,:,0]])
            weather_block_ts = getattr(self, f"weather_conv1d_lrelu_{cl}")(weather_block_ts)
            weather_block_ts = getattr(self, f"weather_conv1x1_{cl}")(weather_block_ts)
            
        weather_block_ts = self.layer_norm_seq_weather(torch.moveaxis(weather_block_ts, 1, -1))
        weather_block_ts = torch.moveaxis(weather_block_ts, -1, 1)
        # Sequential conditioning block
        
        target_ts = target0.unsqueeze(-1)
        
        target_ts = self.cb_transp_conv(target_ts)
        
        target_ts = self.cb_layer_norm(torch.moveaxis(target_ts, 1, -1))
        target_ts = torch.moveaxis(target_ts, -1, 1)
        
        for cl in range(int(self.ccnn_n_layers/2)):
            target_ts = getattr(self, f"cb_conv1d_{cl}")([target_ts, target_ts[:,:,0]])
            target_ts = getattr(self, f"cb_conv1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"cb_conv1x1_{cl}")(target_ts)
            
        target_ts = self.avg_pool_seq_cb(target_ts)
        target_ts = self.layer_norm_seq_cb(torch.moveaxis(target_ts, 1, -1))
        target_ts = torch.moveaxis(target_ts, -1, 1)
            
        # Block join
        
        #target_ts_out = target_ts + weather_block_ts
        
        target_ts_out = torch.cat([target_ts, weather_block_ts], dim = 1)
        
        target_ts_out = self.joint_1x1conv(target_ts_out)
        
        target_ts_out = self.joint_layer_norm_seq(torch.moveaxis(target_ts_out, 1, -1))
        target_ts_out = torch.moveaxis(target_ts_out, -1, 1)
        target_ts_out = self.joint_lrelu_seq(target_ts_out)
        
        
        
        for cl in range(self.ccnn_n_layers):
            target_ts_out = getattr(self, f"joint_conv1d_{cl}")([target_ts_out, target_ts_out[:,:,0]])
            target_ts_out = getattr(self, f"joint_conv1d_lrelu_{cl}")(target_ts_out)
            target_ts_out = getattr(self, f"joint_conv1x1_{cl}")(target_ts_out)
            
        
        target_ts_out = torch.moveaxis(target_ts_out, -1, 1)
        
        target_ts_out = self.fc(target_ts_out).squeeze()
        
        return target_ts_out.squeeze()
    
############################################ MODEL 5 ########################################################

class SC_CCNN_idw(nn.Module):
    
    def __init__(self,
                 timestep = 180,
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
            
        # Joint sequential block
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

############################################ PHYSICS INFORMED ###############################################
############################################ MODEL 1-PI ########################################################

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
                                         nn.Linear(self.ph_params_neurons,
                                                   self.ph_params_neurons),
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
        cb_fc.append(nn.Linear(edim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.ccnn_input_filters))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.ccnn_input_filters)
            
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
        
        self.fc = nn.Linear(self.ccnn_input_filters, 1)

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
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             attn_output.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        target_ts = weather_block_out.squeeze((3,4))
        
        # Sequential block
        for cl in range(self.ccnn_n_layers):
            target_ts = getattr(self, f"convolution_1d_{cl}")([target_ts, target0])
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
        
########################################### MODEL 2-PI #########################################

class SC_PICCNN_att_2(nn.Module):
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
                                         nn.Linear(self.ph_params_neurons,
                                                   self.ph_params_neurons),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.ph_params_neurons, 2))
            
            # self.hydro_cond_lon = torch.FloatTensor([self.ph_params["hyd_cond"][0]])
            # self.hydro_cond_lat = torch.FloatTensor([self.ph_params["hyd_cond"][0]])
            
        # Attention block
        cb_emb_coord = []
        cb_emb_coord.append(nn.Linear(3, self.cb_emb_dim))
        cb_emb_coord.append(nn.LeakyReLU())
        self.cb_emb_coord = nn.Sequential(*cb_emb_coord)
        
        cb_emb_value = []
        cb_emb_value.append(nn.Linear(1, self.cb_emb_dim))
        cb_emb_value.append(nn.LeakyReLU())
        self.cb_emb_value = nn.Sequential(*cb_emb_value)
        
        self.multihead_att_1 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        # q, k, v tutti a 16, sommiamo value e output e concat coord (in 16 dim)
        
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = self.cb_emb_dim*2)
        
        self.fc_att_1 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # q, k, v - q, k sono skip da primi solo coord e v sono valori in output da fc
        
        self.multihead_att_2 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        
        self.layer_norm_3 = nn.LayerNorm(normalized_shape = self.cb_emb_dim*2)
        
        self.fc_att_2 = nn.Sequential(nn.Linear(self.cb_emb_dim*2, self.cb_emb_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.cb_emb_dim, self.cb_emb_dim))
        
        self.layer_norm_4 = nn.LayerNorm(normalized_shape = self.cb_emb_dim)
        
        # qua q = 1, ~ decoder
        # k sono sempre skip coord originarie
        # v dal layer prima
        self.multihead_att_3 = nn.MultiheadAttention(self.cb_emb_dim, self.cb_att_h,
                                                   batch_first=True)
        # concat come vecchio 
        
        self.layer_norm_5 = nn.LayerNorm(normalized_shape = self.cb_emb_dim + 5)
        
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(self.cb_emb_dim + 5, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        
        if self.cb_fc_layer > 2:
            for l in range(self.cb_fc_layer - 2):
                cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
                cb_fc.append(nn.LeakyReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
        cb_fc.append(nn.LeakyReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        self.layer_norm_6 = nn.LayerNorm(normalized_shape= self.cb_fc_neurons)
        
        # Sequental Conditioning Block
        
        self.cb_transp_conv = nn.Sequential(
                                torch.nn.ConvTranspose1d(self.cb_fc_neurons,
                                                        self.cb_fc_neurons,
                                                        16,
                                                        stride=1,
                                                        padding=0,
                                                        bias=True,
                                                        dilation=1,
                                                        ),
                                nn.LeakyReLU(),
                                torch.nn.ConvTranspose1d(self.cb_fc_neurons,
                                                        self.cb_fc_neurons,
                                                        16,
                                                        stride=16,
                                                        padding=0,
                                                        bias=True,
                                                        dilation=1,
                                                        ),
                                nn.LeakyReLU(),
                                nn.Conv1d(self.cb_fc_neurons,
                                        self.ccnn_input_filters,
                                        1,
                                        padding="valid"),
                                nn.AdaptiveAvgPool1d(self.timestep)
                                )
        
        self.cb_layer_norm = nn.LayerNorm(self.ccnn_input_filters)
        
        for cl in range(self.ccnn_n_layers/2):
            setattr(self, f"cb_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               kernel_size= 5,
                                               dilation = 2**cl))
            
            setattr(self, f"cb_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"cb_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        self.layer_norm_seq_cb = nn.LayerNorm(self.ccnn_input_filters)
        
        # Weather block
        self.weather_block = WeatherBlock(conv_filters = self.conv_filters,
                                     output_filters = self.ccnn_input_filters)
            
        
        for cl in range(self.ccnn_n_layers/2):
            setattr(self, f"weather_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               3))
            
            setattr(self, f"weather_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"weather_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        self.layer_norm_seq_weather = nn.LayerNorm(self.ccnn_input_filters)    
        # Joint sequential block
        
        self.joint_1x1conv = nn.Conv1d(self.ccnn_input_filters*2,
                                        self.ccnn_input_filters,
                                        kernel_size= 1,
                                        padding="valid")
        
        self.joint_layer_norm_seq = nn.LayerNorm(int(ccnn_input_filters))
        self.joint_lrelu_seq = nn.LeakyReLU()
        
        
        for cl in range(self.ccnn_n_layers):
            setattr(self, f"joint_conv1d_{cl}",
                    CausalConv1d(self.ccnn_input_filters,
                                               self.ccnn_n_filters,
                                               self.ccnn_kernel_size,
                                               dilation = 2**cl))
            
            setattr(self, f"joint_conv1d_lrelu_{cl}",
                    nn.LeakyReLU())
            
            setattr(self, f"joint_conv1x1_{cl}",
                    nn.Sequential(nn.Conv1d(self.ccnn_n_filters,
                                            self.ccnn_input_filters,
                                            1,
                                            padding="valid"),
                                    nn.LeakyReLU())
                    )
        
        self.fc = nn.Linear(self.ccnn_input_filters, 1)

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
        coords = torch.cat([x[:,:,:3],
                            z.unsqueeze(1)], dim = 1)
        
        cb_emb_coords = self.cb_emb_coord(coords)
        
        query = cb_emb_coords[:,coords.shape[1]-1,:]
        keys = cb_emb_coords[:,:coords.shape[1]-1,:]
        
        cb_emb_values = self.cb_emb_value(x[:,:,-1].unsqueeze(-1))
        
        att_values_1, _ = self.multihead_att_1(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = cb_emb_values,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_1 = att_values_1 + cb_emb_values
        att_values_1 = torch.cat([att_values_1, keys], dim = -1)
        att_values_1 = self.layer_norm_1(att_values_1)
        att_values_1 = self.fc_att_1(att_values_1)
        att_values_1 = self.layer_norm_2(att_values_1)
        
        att_values_2, _ = self.multihead_att_2(
                                            query = keys, #(N,L,E)
                                            key = keys,
                                            value = att_values_1,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        att_values_2 = att_values_2 + att_values_1
        att_values_2 = torch.cat([att_values_2, keys], dim = -1)
        att_values_2 = self.layer_norm_3(att_values_2)
        att_values_2 = self.fc_att_2(att_values_2)
        att_values_2 = self.layer_norm_4(att_values_2)
        
        att_values, attn_output_weights = self.multihead_att_3(
                                            query = query.unsqueeze(1), #(N,L,E)
                                            key = keys,
                                            value = att_values_2,
                                            key_padding_mask = ~x_mask, #(N,S)
                                            )
        
        
        weights_mean = torch.mean(attn_output_weights, dim = (1,2))
        weights_sd = torch.std(attn_output_weights, dim = (1,2))
        
        target0 = torch.cat([z,
                             att_values.squeeze(1),
                             weights_mean.unsqueeze(1),
                             weights_sd.unsqueeze(1)], dim = -1)
        
        target0 = self.layer_norm_5(target0)
        
        target0 = self.cb_fc(target0)
        
        target0 = self.layer_norm_6(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_dist = w[1] - z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_dist = weather_dist[:, None, :, : ,:].expand(-1, w[0].shape[2], -1, -1, -1 )

        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             torch.moveaxis(weather_dist, -1, 1)], dim = 1)
        
        weather_block_out = self.weather_block(weather)
        
        # Sequential block
        weather_block_ts = weather_block_out.squeeze((3,4))
    
        for cl in range(self.ccnn_n_layers/2):
            weather_block_ts = getattr(self, f"weather_conv1d_{cl}")([weather_block_ts, torch.zeros_like(weather_block_ts[:,:,0])])
            weather_block_ts = getattr(self, f"weather_conv1d_lrelu_{cl}")(weather_block_ts)
            weather_block_ts = getattr(self, f"weather_conv1x1_{cl}")(weather_block_ts)
            
        weather_block_ts = self.layer_norm_seq_weather(torch.moveaxis(weather_block_ts, 1, -1))
        weather_block_ts = torch.moveaxis(weather_block_ts, -1, 1)
        # Sequential conditioning block
        
        target_ts = target0.unsqueeze(-1)
        
        target_ts = self.cb_transp_conv(target_ts)
        
        target_ts = self.cb_layer_norm(torch.moveaxis(target_ts, 1, -1))
        target_ts = torch.moveaxis(target_ts, -1, 1)
        
        for cl in range(self.ccnn_n_layers/2):
            target_ts = getattr(self, f"cb_conv1d_{cl}")([target_ts, torch.zeros_like(target_ts[:,:,0])])
            target_ts = getattr(self, f"cb_conv1d_lrelu_{cl}")(target_ts)
            target_ts = getattr(self, f"cb_conv1x1_{cl}")(target_ts)
            
        target_ts = self.layer_norm_seq_cb(torch.moveaxis(target_ts, 1, -1))
        target_ts = torch.moveaxis(target_ts, -1, 1)
            
        # Block join
        
        #target_ts_out = target_ts + weather_block_ts
        
        target_ts_out = torch.cat([target_ts, weather_block_ts], dim = 1)
        
        target_ts_out = self.joint_1x1conv(target_ts_out)
        
        target_ts_out = self.joint_layer_norm_seq(torch.moveaxis(target_ts_out, 1, -1))
        target_ts_out = torch.moveaxis(target_ts_out, -1, 1)
        target_ts_out = self.joint_lrelu_seq(target_ts_out)
        
        
        
        for cl in range(self.ccnn_n_layers):
            target_ts_out = getattr(self, f"joint_conv1d_{cl}")([target_ts_out, torch.zeros_like(target_ts_out[:,:,0])])
            target_ts_out = getattr(self, f"joint_conv1d_lrelu_{cl}")(target_ts_out)
            target_ts_out = getattr(self, f"joint_conv1x1_{cl}")(target_ts_out)
            
        
        target_ts_out = torch.moveaxis(target_ts_out, -1, 1)
        
        target_ts_out = self.fc(target_ts_out).squeeze()
        
        # Physics Block
        hyd_cond = self.ph_params_fc(z)
        hyd_cond = torch.clamp(hyd_cond,
                                min=self.ph_params["hyd_cond"][1],
                                max=self.ph_params["hyd_cond"][2])
        
        hyd_cond = hyd_cond + (torch.ones_like(hyd_cond)*self.ph_params["hyd_cond"][0])
            
        if hc_out is True:    
            return target_ts_out, hyd_cond
        
        else: 
            return target_ts_out
        

###########################################
    
if __name__ == "__main__":
    # print("Loading data.json...")
    # config = {}
    # with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
    #     config = json.load(f)
    # print(f"Read data.json: {config}")

    # print("Loading ContinuousDataset...")
    # ds = Dataset_1D(config)
    # x, z, w_values, y, x_mask, y_mask  = ds[0]
    
    # weather_coords = ds.get_weather_coords()
    # weather_dtm = ds.get_weather_dtm()
    # weather_coords = torch.cat([weather_coords, weather_dtm], dim = -1)
    # weather_coords_batch = weather_coords.unsqueeze(0)
    # w = [w_values, weather_coords_batch]
    
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )

    # print("Loading Continuous1DNN...")
    # timesteps = config["timesteps"]
    # model = SC_LSTM_idw(timestep = config["timesteps"],
    #              cb_fc_layer = config["cb_fc_layer"], #5,
    #              cb_fc_neurons = config["cb_fc_neurons"], # 32,
    #              conv_filters = config["conv_filters"], #32,
    #              lstm_layer = config["lstm_layer"], #5,
    #              lstm_input_units = config["lstm_input_units"], #16,
    #              lstm_units = config["lstm_units"] #32
    #              ).to(device)
    # print("Continuous1DNN prediction...")
    # y = model(x, z, w, x_mask)
    # print(f"Output:\n{y}")
    
    pass
    