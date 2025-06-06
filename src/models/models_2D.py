from functools import partial
import copy
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.dataset_1d import Dataset_1D

##### Blocks ######

def densification_dropout(sample, p = 0.25):
        
        """
        Dropout training as in densification of Andrychowicz et al. (2023)
        """
        
        new_X = sample[0].clone()
        new_X_mask = sample[1].clone()
        
        for batch in range(new_X.shape[0]):
            avail_X = new_X_mask[batch,:].nonzero().squeeze()
            dropout = torch.randint(0, len(avail_X), [int(len(avail_X)*p)])
            new_X_mask[batch,avail_X[dropout]] = False
            new_X[batch,avail_X[dropout],-1] = 0
        
        return new_X, new_X_mask
    

class MoveAxis(nn.Module):
    
    def __init__(self, source, destination):
        super().__init__()
        
        self.moveaxis = partial(torch.moveaxis, source = source, destination = destination)
    
    def forward(self, x):
        return self.moveaxis(x)

class LayerNorm_MA(nn.Module):
    
    def __init__(self, normalized_shape, move_dim_from = None, move_dim_to = None, 
                 eps = 0.00001, elementwise_affine = True, bias = True, device=None, dtype=None):
        super().__init__()
        
        self.norm = []
        self.norm.append(nn.LayerNorm(normalized_shape, elementwise_affine=elementwise_affine, eps = eps,
                                      bias = bias, device = device, dtype = dtype))
        
        if move_dim_from is not None:
            self.norm.insert(0, MoveAxis(move_dim_from, move_dim_to)) #partial(torch.moveaxis, source = norm_dim, destination = -1)
            self.norm.append(MoveAxis(move_dim_to, move_dim_from)) #partial(torch.moveaxis, source = -1, destination = norm_dim)
            
        self.norm = nn.Sequential(*self.norm)
        
    def forward(self, input):
            
        norm_output = self.norm(input)
        return norm_output
        
        

class Weather_SubPixelUpsampling_Block(nn.Module):
    
    def compute_3dTrConv_out_dim(self, h_in, k_size):
        h_out = (h_in - 1) + (k_size - 1) + 1
        return h_out
    
    def __init__(self,
                 input_dimensions = [10, 9, 12],
                 hidden_channels = 32,
                 output_channels = 16,
                 output_dim = [104, 150],
                 padding_mode = "replicate",
                 layernorm_affine = False):
        super().__init__()
        
        self.subpixel_upsampling = []
        
        self.subpixel_upsampling.append(nn.Conv3d(input_dimensions[0], hidden_channels, (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.subpixel_upsampling.append(LayerNorm_MA(hidden_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.subpixel_upsampling.append(nn.LeakyReLU())
        self.subpixel_upsampling.append(nn.Conv3d(hidden_channels, hidden_channels, (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.subpixel_upsampling.append(LayerNorm_MA(hidden_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.subpixel_upsampling.append(nn.LeakyReLU())

        
        self.subpixel_upsampling.append(nn.Conv3d(int(hidden_channels), int(6*6*hidden_channels), (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        #self.subpixel_upsampling.append(LayerNorm_MA(int(6*6*output_channels), move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.subpixel_upsampling.append(nn.LeakyReLU())
        
        self.subpixel_upsampling.append(MoveAxis(source = 1, destination = 2))
        
        self.subpixel_upsampling.append(nn.PixelShuffle(3))
        
        self.subpixel_upsampling.append(MoveAxis(source = 2, destination = 1))
        
        self.subpixel_upsampling.append(nn.Conv3d(hidden_channels, output_channels, (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.subpixel_upsampling.append(LayerNorm_MA(output_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.subpixel_upsampling.append(nn.LeakyReLU())
        
        self.subpixel_upsampling.append(nn.AdaptiveAvgPool3d((None,output_dim[0],output_dim[1])))
        
        self.subpixel_upsampling.append(nn.Conv3d(output_channels, output_channels, (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.subpixel_upsampling.append(LayerNorm_MA(output_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.subpixel_upsampling.append(nn.LeakyReLU())
        
        self.subpixel_upsampling = nn.Sequential(*self.subpixel_upsampling)
    
    def forward(self, x):
        
        return self.subpixel_upsampling(x)

class Weather_Upsampling_Block(nn.Module):
    
    def compute_3dTrConv_out_dim(self, h_in, k_size):
        h_out = (h_in - 1) + (k_size - 1) + 1
        return h_out
    
    def __init__(self,
                 input_dimensions = [10, 9, 12],
                 hidden_channels = 32,
                 output_channels = 16,
                 output_dim = [104, 150],
                 padding_mode = "replicate",
                 layernorm_affine = False):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Conv3d(input_dimensions[0], hidden_channels, (1,3,3), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.layers.append(LayerNorm_MA(hidden_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Conv3d(hidden_channels, hidden_channels, (1,3,3), padding='same', padding_mode = padding_mode, dtype=torch.float32))
        self.layers.append(LayerNorm_MA(hidden_channels, move_dim_from = 1, move_dim_to = -1, elementwise_affine = layernorm_affine))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.AdaptiveAvgPool3d((None,output_dim[0],output_dim[1])))
        
        self.layers.append(nn.Conv3d(hidden_channels, output_channels, (1,5,5), padding='same', padding_mode = padding_mode, dtype=torch.float32))        
        self.layers.append(nn.LeakyReLU())
        
        self.block = nn.Sequential(*self.layers)
    
    def forward(self, input):
        return self.block(input)
    
class Weather_Attention_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 input_channles,
                 heads,
                 output_dims):
        super().__init__()
        
        self.output_dims = output_dims
        
        cb_topo_emb = []
        cb_topo_emb.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
        cb_topo_emb.append(nn.GELU())
        self.cb_topo_emb = nn.Sequential(*cb_topo_emb)
        
        cb_value_emb = []
        cb_value_emb.append(nn.Linear(input_channles, embedding_dim))
        cb_value_emb.append(nn.GELU())
        self.cb_value_emb = nn.Sequential(*cb_value_emb)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear_1 = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = embedding_dim),
                                    nn.Linear(embedding_dim, output_dims[0]),
                                    nn.GELU(),
                                    )
        
        # self.cb_multihead_att_2 = nn.MultiheadAttention(embedding_dim, heads,
        #                                            batch_first=True)
        
        # self.norm_linear_2 = nn.Sequential(
        #                             nn.LayerNorm(normalized_shape = embedding_dim),
        #                             nn.Linear(embedding_dim, embedding_dim),
        #                             nn.LeakyReLU(),
        #                             )
        
        # self.cb_multihead_att_3 = nn.MultiheadAttention(embedding_dim, heads,
        #                                            batch_first=True)
        
        # self.norm_linear_3 = nn.Sequential(
        #                             nn.LayerNorm(normalized_shape = embedding_dim),
        #                             nn.Linear(embedding_dim, output_dims[0]),
        #                             nn.LeakyReLU(),
        #                             )
        
    def forward(self, K, V, Q):
            
            coords = torch.cat([K,
                                Q],
                               dim = 1)
            
            topographical_embedding = self.cb_topo_emb(coords)
            
            keys = topographical_embedding[:,:K.shape[1],:]
            queries = topographical_embedding[:,K.shape[1]:,:]
            values = self.cb_value_emb(V)
            
            weather_out, _ = self.cb_multihead_att_1(
                                            query = queries, #(N,L,E)
                                            key = keys,
                                            value = values
                                            )
            weather_out = self.norm_linear_1(weather_out)
            
            # weather_out_skip_2 = weather_out
            # weather_out, _ = self.cb_multihead_att_2(
            #                                 query = weather_out, #(N,L,E)
            #                                 key = weather_out,
            #                                 value = weather_out
            #                                 )
            # weather_out = weather_out + weather_out_skip_2
            # weather_out = self.norm_linear_2(weather_out)
            
            # weather_out_skip_3 = weather_out
            # weather_out, _ = self.cb_multihead_att_3(
            #                                 query = weather_out, #(N,L,E)
            #                                 key = weather_out,
            #                                 value = weather_out
            #                                 )
            # weather_out = weather_out + weather_out_skip_3
            # weather_out = self.norm_linear_3(weather_out)
            
            
            
            weather_out = torch.moveaxis(weather_out, -1, 1)
            weather_out = torch.reshape(weather_out, (*weather_out.shape[:2],
                                         self.output_dims[1], self.output_dims[2]))

            return weather_out

class Conditioning_Attention_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 heads,
                 output_channels):
        super().__init__()
        
        cb_topo_emb = []
        cb_topo_emb.append(nn.Linear(3, embedding_dim))
        cb_topo_emb.append(nn.GELU())
        self.cb_topo_emb = nn.Sequential(*cb_topo_emb)
        
        cb_value_emb = []
        cb_value_emb.append(nn.Linear(4, embedding_dim))
        cb_value_emb.append(nn.GELU())
        self.cb_value_emb = nn.Sequential(*cb_value_emb)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear_1 = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = embedding_dim),
                                    nn.Linear(embedding_dim, output_channels),
                                    nn.GELU(),
                                    )
        
        # self.cb_multihead_att_2 = nn.MultiheadAttention(embedding_dim, heads,
        #                                            batch_first=True)
        
        # self.norm_linear_2 = nn.Sequential(
        #                             nn.LayerNorm(normalized_shape = embedding_dim),
        #                             nn.Linear(embedding_dim, embedding_dim),
        #                             nn.LeakyReLU(),
        #                             )
        
        # self.cb_multihead_att_3 = nn.MultiheadAttention(embedding_dim, heads,
        #                                            batch_first=True)
        
        # self.norm_linear_3 = nn.Sequential(
        #                             nn.LayerNorm(normalized_shape = embedding_dim),
        #                             nn.Linear(embedding_dim, output_channels),
        #                             nn.LeakyReLU(),
        #                             )
        
    def forward(self, X, Z, X_mask):
            
            coords = torch.cat([X[:,:,:3],
                                Z.flatten(start_dim = 1, end_dim = 2)],
                               dim = 1)
            
            topographical_embedding = self.cb_topo_emb(coords)
            
            keys = topographical_embedding[:,:X.shape[1],:]
            queries = topographical_embedding[:,X.shape[1]:,:]
            values = self.cb_value_emb(X)
            
            target_Icond, _ = self.cb_multihead_att_1(
                                            query = queries, #(N,L,E)
                                            key = keys,
                                            value = values,
                                            key_padding_mask = ~X_mask, #(N,S)
                                            )
            
            target_Icond = self.norm_linear_1(target_Icond)
            
            # target_Icond_skip_2 = target_Icond
            # target_Icond, _ = self.cb_multihead_att_2(
            #                                 query = target_Icond, #(N,L,E)
            #                                 key = target_Icond,
            #                                 value = target_Icond
            #                                 )
            # target_Icond = target_Icond + target_Icond_skip_2
            # target_Icond = self.norm_linear_2(target_Icond)
            
            # target_Icond_skip_3 = target_Icond
            # target_Icond, _ = self.cb_multihead_att_3(
            #                                 query = target_Icond, #(N,L,E)
            #                                 key = target_Icond,
            #                                 value = target_Icond
            #                                 )
            # target_Icond = target_Icond + target_Icond_skip_3
            # target_Icond = self.norm_linear_3(target_Icond)
            
            target_Icond = torch.moveaxis(target_Icond, -1, 1)
            target_Icond = torch.reshape(target_Icond, (*target_Icond.shape[:2],
                                         Z.shape[1], Z.shape[2]))

            return target_Icond

# Inspired by https://github.com/czifan/ConvLSTM.pytorch/blob/master/networks/ConvLSTM.py (to cite)
class ConvLSTMBlock(nn.Module):
    def __init__(self, 
                 input_channles,
                 hidden_channels,
                 kernel_size=5,
                 padding="same",
                 stride=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = self._make_layer(input_channles+hidden_channels, hidden_channels*4,
                                       kernel_size, padding, stride)
        
        self.value_activation = nn.Tanh() #nn.Tanh() #nn.LeakyReLU()
        

    def _make_layer(self, input_channles, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(input_channles, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=True, padding_mode="replicate"),
            #LayerNorm_MA((out_channels,*self.HW_dimensions))
            #nn.BatchNorm2d(out_channels)
            )

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

            cy = (forgetgate * c_0) + (ingate * self.value_activation(cellgate))
            hy = outgate * self.value_activation(cy)
            outputs.append(hy)
            h_0 = hy
            c_0 = cy

        return [torch.stack(outputs, dim = 2), (hy, cy)] # (N, C, D, H, W)
    
    
class Date_Conditioning_Block(nn.Module):
    def __init__(self,
                 n_channels):
        super().__init__()
        
        self.n_channels = n_channels
        
        for i in range(n_channels):
            
            setattr(self, f"fc_0_{i}",
                    nn.Linear(2, 16))
            setattr(self, f"activation_0_{i}",
                    nn.GELU()),
            setattr(self, f"fc_1_{i}",
                    nn.Linear(16, 2))
            # setattr(self, f"activation_1_{i}",
            #         nn.LeakyReLU()),
            
            
    def forward(self, input):
        outputs = []
        for i in range(self.n_channels):
            output = getattr(self, f"fc_0_{i}")(input)
            output = getattr(self, f"activation_0_{i}")(output)
            output = getattr(self, f"fc_1_{i}")(output)
            # output = getattr(self, f"activation_1_{i}")(output)
            
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim = 1)
        
        return outputs


class Date_Conditioning_Block_alt(nn.Module):
    def __init__(self,
                 hidden_dimension):
        super().__init__()
        
        self.hidden_dimension = hidden_dimension
        
        self.date_conditioning = nn.Sequential(nn.Linear(2, self.hidden_dimension),
                                               nn.LeakyReLU())
            
            
    def forward(self, input):
        
        output = self.date_conditioning(input)
        
        return output
        
        

########## ImageConditioning #################
    
class AttCB_ConvLSTM(nn.Module):
    def __init__(self,
                 weather_CHW_dim = [10, 9, 12],
                 cb_emb_dim = 16,
                 cb_heads = 4,
                 channels_cb_wb = 32,
                 convlstm_input_units = 16,
                 convlstm_units = 32,
                 convlstm_kernel = 5,
                 densification_dropout = 0.5,
                 upsampling_dim = [104, 150],
                 layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        self.convlstm_input_units = convlstm_input_units
        self.convlstm_units = convlstm_units
        self.convlstm_kernel = convlstm_kernel
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb_wb = channels_cb_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 hidden_channels = self.channels_cb_wb,
                 output_channels = self.convlstm_units, #self.convlstm_units,
                 densification_dropout_p = self.densification_dropout_p)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(self.convlstm_input_units)
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Upsampling_Block(input_dimensions = self.input_dimension,
                 hidden_channels = self.channels_cb_wb,
                 output_channels = self.convlstm_input_units,
                 output_dim = self.upsampling_dim)
        
        ### Sequential module ###
        
        self.convLSTM_1 = ConvLSTMBlock(input_channles = int(self.convlstm_input_units),
                 hidden_channels = self.convlstm_units,
                 HW_dimensions = self.upsampling_dim,
                 kernel_size=self.convlstm_kernel)
        
        self.Date_Conditioning_Module_sm = Date_Conditioning_Block(self.convlstm_units)
        self.Layer_Norm = LayerNorm_MA([self.convlstm_units,*self.upsampling_dim], 1, 2, elementwise_affine = layernorm_affine)
        
        self.convLSTM_2 =ConvLSTMBlock(input_channles = self.convlstm_units,
                 hidden_channels = self.convlstm_units,
                 HW_dimensions= self.upsampling_dim,
                 kernel_size=self.convlstm_kernel)
        
        self.convLSTM_3 =ConvLSTMBlock(input_channles = self.convlstm_units,
                 hidden_channels = self.convlstm_units,
                 HW_dimensions= self.upsampling_dim,
                 kernel_size=self.convlstm_kernel)
        
        self.linear = nn.Linear(int(self.convlstm_units), 1)
        
    def forward(self, X, Z, W, X_mask):
        
        ### Conditioning modules ###
        
        target_Icond = self.Icondition_Module(X, Z, X_mask)
        
        ### Weather module ### 
        
        Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        
        # Weaether_seq = torch.cat([Weaether_seq,
        #                           target_Icond.unsqueeze(2).expand(-1,-1,Weaether_seq.shape[2],-1,-1)], 
        #                          dim = 1)
        
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # N, C, D, 2
        date_conditioning_wm = date_conditioning_wm[:,:,:,None, None,:].expand(-1, -1, -1,
                                                                         Weaether_seq.shape[3],
                                                                         Weaether_seq.shape[4],
                                                                         -1)
        
        Weaether_seq = (Weaether_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
        
        ### Sequential module ### 
        
        Output = self.convLSTM_1(Weaether_seq,
                                 target_Icond,
                                 target_Icond
                                 )
        
        date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        date_conditioning_sm = date_conditioning_sm[:,:,:,None, None,:].expand(-1, -1, -1,
                                                                         Output.shape[3],
                                                                         Output.shape[4],
                                                                         -1)
        
        Output = (Output * date_conditioning_sm[:,:,:,:,:,0]) + date_conditioning_sm[:,:,:,:,:,1]
        Output = self.Layer_Norm(Output)
        
        Output = self.convLSTM_2(Output,
                                 target_Icond,
                                 target_Icond
                                 )
        
        Output = self.convLSTM_3(Output,
                                 target_Icond,
                                 target_Icond
                                 )
        
        Output = self.linear(torch.moveaxis(Output, 1, -1))
        #Output = self.linear_2(Output)
        
        Output = torch.moveaxis(Output, -1, 1)
        
        return Output.squeeze()

##############################################
######### VideoConditioning ##################
##############################################

class VideoCB_ConvLSTM(nn.Module):
    def __init__(self,
                 weather_CHW_dim = [10, 9, 12],
                 cb_emb_dim = 16,
                 cb_heads = 4,
                 channels_cb = 32,
                 channels_wb = 32,
                 convlstm_IO_units = 16,
                 convlstm_hidden_units = 32,
                 convlstm_nlayer = 3,
                 convlstm_kernel = 5,
                 densification_dropout = 0.5,
                 upsampling_dim = [104, 150],
                 spatial_dropout = 0.35,
                 layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        self.convlstm_IO_units = convlstm_IO_units
        self.convlstm_hidden_units = convlstm_hidden_units
        self.convlstm_kernel = convlstm_kernel
        self.convlstm_nlayer = convlstm_nlayer
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb = channels_cb
        self.channels_wb = channels_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 hidden_channels = self.channels_cb,
                 output_channels = self.convlstm_IO_units, #self.convlstm_units,
                 layernorm_affine = self.layernorm_affine)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(int(self.convlstm_IO_units*2))
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Upsampling_Block(input_dimensions = self.input_dimension,
                 hidden_channels = self.channels_wb,
                 output_channels = self.convlstm_IO_units,
                 output_dim = self.upsampling_dim,
                 layernorm_affine = self.layernorm_affine)
        
        ### Join Modoule ### 
        
        self.Joint_Conv3d = nn.Sequential(nn.Conv3d(int(self.convlstm_IO_units*2),
                                                self.convlstm_IO_units,
                                                kernel_size=(1,5,5), padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(self.convlstm_IO_units, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    nn.LeakyReLU())
        
        input_features = self.convlstm_IO_units
        
        for i in range(self.convlstm_nlayer):
                
            setattr(self, f"convLSTM_{i}",
                    ConvLSTMBlock(input_channles = input_features,
                                    hidden_channels = self.convlstm_hidden_units[i],
                                    kernel_size=self.convlstm_kernel[i]))
            
            input_features = self.convlstm_hidden_units[i]
            
        
        self.Output_layer = nn.Sequential(
                                        MoveAxis(1,-1),
                                        nn.Linear(self.convlstm_hidden_units[-1], self.convlstm_IO_units),
                                        LayerNorm_MA(self.convlstm_IO_units,
                                                     elementwise_affine = self.layernorm_affine),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.convlstm_IO_units, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
            
        Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # alt N, D, C  # N, C, D, 2
        #date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Target_VideoCond = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                Target_VideoCond.append(self.Icondition_Module(X_t, Z, X_mask_t))
                
            Target_VideoCond = torch.stack(Target_VideoCond, dim = 2)
            
            date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                Weaether_seq.shape[-2],
                                                                                Weaether_seq.shape[-1],
                                                                                -1)
            # print(Weaether_seq.shape)
            # print(Target_VideoCond.shape)
            Joint_seq = torch.cat([Weaether_seq,
                                    Target_VideoCond], 
                                    dim = 1)
            
            Joint_seq = (Joint_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
            
            
            Joint_seq = self.Joint_Conv3d(Joint_seq)
            
            #Joint_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)
            
            ### Sequential module ### 
            
            for i in range(self.convlstm_nlayer):
                
                Joint_seq, _ = getattr(self, f"convLSTM_{i}")(Joint_seq,
                                                               #ConvLSTM_hidden_state,
                                                               #ConvLSTM_hidden_state
                                                               )
            
            Output_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)          
            
            Output = self.Output_layer(Output_seq)
            
            return Output.squeeze()
        
        else:
            
            ImageCond = X
            ImageCond_mask = X_mask
            Output = []
            
            convlstm_h_state = [None for i in range(self.convlstm_nlayer)]
            convlstm_c_state = [None for i in range(self.convlstm_nlayer)]
            
            for timestep in range(W[0].shape[2]):
                
                Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask)
                
                
                date_conditioning_wm_Image = date_conditioning_wm[:,:,timestep,:]
                
                
                date_conditioning_wm_Image = date_conditioning_wm_Image[:,:,None,None,:].expand(-1,-1,
                                                                                Weaether_seq.shape[-2],
                                                                                Weaether_seq.shape[-1],
                                                                                -1)
                
                Joint_Image = torch.cat([Weaether_seq[:,:,timestep,:,:],
                                    Upsampled_ImageCond], 
                                    dim = 1)
                
                Joint_Image = (Joint_Image * date_conditioning_wm_Image[:,:,:,:,0]) + date_conditioning_wm_Image[:,:,:,:,1]
                Joint_Image = Joint_Image.unsqueeze(2)
                
                Joint_Image = self.Joint_Conv3d(Joint_Image)
                
                # Spatial Dropout
                #Joint_Image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
            
                ### Sequential module ###         
                for i in range(self.convlstm_nlayer):
                    
                    Joint_Image, (convlstm_h_state[i], convlstm_c_state[i]) = getattr(self, f"convLSTM_{i}")(Joint_Image,
                                                                                                              convlstm_h_state[i],
                                                                                                              convlstm_c_state[i])
                
                
                # Spatial Dropout
                Output_image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
                
                Output_image = self.Output_layer(Output_image)
            
                ImageCond = torch.cat([Z.clone(),
                                       Output_image[:,0,:,:,:]],
                                      dim=-1)
                ImageCond = ImageCond.flatten(start_dim = 1, end_dim = 2)
                ImageCond_mask = torch.ones((ImageCond[:,:,0].shape)).to(torch.bool).to(ImageCond.device)
            
                Output.append(Output_image)
                
        
        Output = torch.cat(Output, dim=1).squeeze()
            
        return Output
        

class VideoCB_ConvLSTM_PI(nn.Module):
    def __init__(self,
                 weather_CHW_dim = [10, 9, 12],
                 cb_emb_dim = 16,
                 cb_heads = 4,
                 channels_cb = 32,
                 channels_wb = 32,
                 convlstm_IO_units = 16,
                 convlstm_hidden_units = 32,
                 convlstm_nlayer = 3,
                 convlstm_kernel = 5,
                 densification_dropout = 0.5,
                 upsampling_dim = [104, 150],
                 spatial_dropout = 0.35,
                 layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        self.convlstm_IO_units = convlstm_IO_units
        self.convlstm_hidden_units = convlstm_hidden_units
        self.convlstm_kernel = convlstm_kernel
        self.convlstm_nlayer = convlstm_nlayer
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb = channels_cb
        self.channels_wb = channels_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 output_channels = self.convlstm_IO_units, #self.convlstm_units
                 )
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(int(self.convlstm_IO_units*2))
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Upsampling_Block(input_dimensions = self.input_dimension,
                 hidden_channels = self.channels_wb,
                 output_channels = self.convlstm_IO_units,
                 output_dim = self.upsampling_dim,
                 layernorm_affine = self.layernorm_affine)
        
        ### Join Modoule ### 
        
        self.Joint_Conv3d = nn.Sequential(nn.Conv3d(int(self.convlstm_IO_units*2),
                                                self.convlstm_IO_units,
                                                kernel_size=(1,5,5), padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(self.convlstm_IO_units, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    nn.LeakyReLU())
        
        input_features = self.convlstm_IO_units
        
        for i in range(self.convlstm_nlayer):
                
            setattr(self, f"convLSTM_{i}",
                    ConvLSTMBlock(input_channles = input_features,
                                    hidden_channels = self.convlstm_hidden_units[i],
                                    kernel_size=self.convlstm_kernel[i]))
            
            input_features = self.convlstm_hidden_units[i]
            
        
        self.Output_layer = nn.Sequential(
                                        MoveAxis(1,-1),
                                        nn.Linear(self.convlstm_hidden_units[-1], self.convlstm_IO_units),
                                        LayerNorm_MA(self.convlstm_IO_units,
                                                     elementwise_affine = self.layernorm_affine),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.convlstm_IO_units, 1))
        
        self.K_layer = nn.Sequential(
                                    MoveAxis(-1,1),
                                    nn.Conv2d(3,
                                                32,
                                                kernel_size=5, padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(32, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32,
                                                2,
                                                kernel_size=1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False, K_out = False):
        
        ### Weather module ### 
            
        Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # alt N, D, C  # N, C, D, 2
        #date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Target_VideoCond = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                Target_VideoCond.append(self.Icondition_Module(X_t, Z, X_mask_t))
                
            Target_VideoCond = torch.stack(Target_VideoCond, dim = 2)
            
            date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                Weaether_seq.shape[-2],
                                                                                Weaether_seq.shape[-1],
                                                                                -1)
            # print(Weaether_seq.shape)
            # print(Target_VideoCond.shape)
            Joint_seq = torch.cat([Weaether_seq,
                                    Target_VideoCond], 
                                    dim = 1)
            
            Joint_seq = (Joint_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
            
            
            Joint_seq = self.Joint_Conv3d(Joint_seq)
            
            #Joint_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)
            
            ### Sequential module ### 
            
            for i in range(self.convlstm_nlayer):
                
                Joint_seq, _ = getattr(self, f"convLSTM_{i}")(Joint_seq,
                                                               #ConvLSTM_hidden_state,
                                                               #ConvLSTM_hidden_state
                                                               )
            
            Output_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)          
            
            Output = self.Output_layer(Output_seq)
            
            if K_out is True:
                
                    K_lat_lon = self.K_layer(Z)
                    K_lat_lon = torch.clamp(K_lat_lon,
                                            min = 1e-2,
                                            max = 1e2)
            
                    return [Output.squeeze(), K_lat_lon]
                
            else:
                return Output.squeeze()
        
        else:
            
            ImageCond = X
            ImageCond_mask = X_mask
            Output = []
            
            convlstm_h_state = [None for i in range(self.convlstm_nlayer)]
            convlstm_c_state = [None for i in range(self.convlstm_nlayer)]
            
            for timestep in range(W[0].shape[2]):
                
                Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask)
                
                
                date_conditioning_wm_Image = date_conditioning_wm[:,:,timestep,:]
                
                
                date_conditioning_wm_Image = date_conditioning_wm_Image[:,:,None,None,:].expand(-1,-1,
                                                                                Weaether_seq.shape[-2],
                                                                                Weaether_seq.shape[-1],
                                                                                -1)
                
                Joint_Image = torch.cat([Weaether_seq[:,:,timestep,:,:],
                                    Upsampled_ImageCond], 
                                    dim = 1)
                
                Joint_Image = (Joint_Image * date_conditioning_wm_Image[:,:,:,:,0]) + date_conditioning_wm_Image[:,:,:,:,1]
                Joint_Image = Joint_Image.unsqueeze(2)
                
                Joint_Image = self.Joint_Conv3d(Joint_Image)
                
                # Spatial Dropout
                #Joint_Image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
            
                ### Sequential module ###         
                for i in range(self.convlstm_nlayer):
                    
                    Joint_Image, (convlstm_h_state[i], convlstm_c_state[i]) = getattr(self, f"convLSTM_{i}")(Joint_Image,
                                                                                                              convlstm_h_state[i],
                                                                                                              convlstm_c_state[i])
                
                
                # Spatial Dropout
                Output_image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
                
                Output_image = self.Output_layer(Output_image)
            
                ImageCond = torch.cat([Z.clone(),
                                       Output_image[:,0,:,:,:]],
                                      dim=-1)
                ImageCond = ImageCond.flatten(start_dim = 1, end_dim = 2)
                ImageCond_mask = torch.ones((ImageCond[:,:,0].shape)).to(torch.bool).to(ImageCond.device)
            
                Output.append(Output_image)
                
                
                #torch.moveaxis(target_Icond, -1, 1)
        # torch.clamp(hyd_cond,
        #                         min=self.ph_params["hyd_cond"][1],
        #                         max=self.ph_params["hyd_cond"][2])
                
                
        
        Output = torch.cat(Output, dim=1).squeeze()
        
        if K_out is True:
                
                    K_lat_lon = self.K_layer(Z)
                    K_lat_lon = torch.clamp(K_lat_lon,
                                            min = 1e-4,
                                            max = 1e3)
            
                    return [Output, K_lat_lon]
                
        else:
            return Output


class FullAttention_ConvLSTM(nn.Module):
    def __init__(self,
                 weather_CHW_dim = [10, 9, 12],
                 cb_emb_dim = 16,
                 cb_heads = 4,
                 channels_cb = 32,
                 channels_wb = 32,
                 convlstm_IO_units = 16,
                 convlstm_hidden_units = 32,
                 convlstm_nlayer = 3,
                 convlstm_kernel = 5,
                 densification_dropout = 0.5,
                 upsampling_dim = [104, 150],
                 spatial_dropout = 0.35,
                 layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        self.convlstm_IO_units = convlstm_IO_units
        self.convlstm_hidden_units = convlstm_hidden_units
        self.convlstm_kernel = convlstm_kernel
        self.convlstm_nlayer = convlstm_nlayer
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb = channels_cb
        self.channels_wb = channels_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 output_channels = self.convlstm_IO_units)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(int(self.convlstm_IO_units*2))
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Attention_Block(
                 embedding_dim = self.cb_emb_dim,
                 input_channles = self.input_dimension[0],
                 heads = self.cb_heads,
                 output_dims = [self.convlstm_IO_units, *self.upsampling_dim])
        
        ### Join Modoule ### 
        
        self.Joint_Conv3d = nn.Sequential(nn.Conv3d(int(self.convlstm_IO_units*2),
                                                self.convlstm_IO_units,
                                                kernel_size=(1,5,5), padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(self.convlstm_IO_units, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    nn.LeakyReLU())
        
        input_features = self.convlstm_IO_units
        
        for i in range(self.convlstm_nlayer):
            
            # setattr(self, f"HiddenState_convLSTM_{i}",
            #         nn.Sequential(nn.Conv2d(self.convlstm_IO_units,
            #                   self.convlstm_hidden_units[i],
            #                   kernel_size=1),
            #         nn.Tanh()))
                
            setattr(self, f"convLSTM_{i}",
                    ConvLSTMBlock(input_channles = input_features,
                                    hidden_channels = self.convlstm_hidden_units[i],
                                    kernel_size=self.convlstm_kernel[i]))
            
            input_features = self.convlstm_hidden_units[i]
            
        
        self.Output_layer = nn.Sequential(
                                        MoveAxis(1,-1),
                                        nn.Linear(self.convlstm_hidden_units[-1], self.convlstm_IO_units),
                                        LayerNorm_MA(self.convlstm_IO_units,
                                                     elementwise_affine = self.layernorm_affine),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.convlstm_IO_units, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
            
        #Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # alt N, D, C  # N, C, D, 2
        #date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Upsampled_VideoCond = []
            Upsampled_VideoWeather = []
            # Target_VideoCond = []
            # Upsampled_VideoWeather = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                Upsampled_VideoCond.append(self.Icondition_Module(X_t, Z, X_mask_t))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_VideoWeather.append(self.Weather_Module(Weather_Keys,
                                                                       Weather_Values,
                                                                       Z.flatten(start_dim = 1, end_dim = 2)
                                                                       )) # K, V, Q
                
            Upsampled_VideoCond = torch.stack(Upsampled_VideoCond, dim = 2)
            Upsampled_VideoWeather = torch.stack(Upsampled_VideoWeather, dim = 2)
            
            date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                self.upsampling_dim[0],
                                                                                self.upsampling_dim[1],
                                                                                -1)
            # print(Weaether_seq.shape)
            # print(Target_VideoCond.shape)
            Joint_seq = torch.cat([Upsampled_VideoWeather,
                                    Upsampled_VideoCond], 
                                    dim = 1)
            
            Joint_seq = (Joint_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
            
            
            Joint_seq = self.Joint_Conv3d(Joint_seq)
            
            #Joint_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)
            
            ### Sequential module ### 
            
            for i in range(self.convlstm_nlayer):
                
                
                #ConvLSTM_hidden_state = getattr(self, f"HiddenState_convLSTM_{i}")(Upsampled_VideoCond[:,:,0,:,:])
                
                Joint_seq, _ = getattr(self, f"convLSTM_{i}")(Joint_seq,
                                                            #    ConvLSTM_hidden_state,
                                                            #    ConvLSTM_hidden_state
                                                               )
            
            Output_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)          
            
            Output = self.Output_layer(Output_seq)
            
            return Output.squeeze()
        
        else:
            
            ImageCond = X
            ImageCond_mask = X_mask
            Output = []
            
            convlstm_h_state = [None for i in range(self.convlstm_nlayer)]
            convlstm_c_state = [None for i in range(self.convlstm_nlayer)]
            
            for timestep in range(W[0].shape[2]):
                
                Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_ImageWeather = self.Weather_Module(Weather_Keys,
                                                        Weather_Values,
                                                        Z.flatten(start_dim = 1, end_dim = 2)
                                                        )
                
                date_conditioning_wm_Image = date_conditioning_wm[:,:,timestep,:]
                
                
                date_conditioning_wm_Image = date_conditioning_wm_Image[:,:,None,None,:].expand(-1,-1,
                                                                                self.upsampling_dim[0],
                                                                                self.upsampling_dim[1],
                                                                                -1)
                
                Joint_Image = torch.cat([Upsampled_ImageWeather,
                                    Upsampled_ImageCond], 
                                    dim = 1)
                
                Joint_Image = (Joint_Image * date_conditioning_wm_Image[:,:,:,:,0]) + date_conditioning_wm_Image[:,:,:,:,1]
                Joint_Image = Joint_Image.unsqueeze(2)
                
                Joint_Image = self.Joint_Conv3d(Joint_Image)
                
                # Spatial Dropout
                #Joint_Image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
            
                ### Sequential module ###         
                for i in range(self.convlstm_nlayer):
                    
                    # if timestep == 0:
                    #     convlstm_h_state[i] = getattr(self, f"HiddenState_convLSTM_{i}")(Upsampled_ImageCond)
                    #     convlstm_c_state[i] = convlstm_h_state[i]
                        
                    
                    Joint_Image, (convlstm_h_state[i], convlstm_c_state[i]) = getattr(self, f"convLSTM_{i}")(Joint_Image,
                                                                                                              convlstm_h_state[i],
                                                                                                              convlstm_c_state[i])
                
                
                # Spatial Dropout
                Output_image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
                
                Output_image = self.Output_layer(Output_image)
            
                ImageCond = torch.cat([Z.clone(),
                                       Output_image[:,0,:,:,:]],
                                      dim=-1)
                ImageCond = ImageCond.flatten(start_dim = 1, end_dim = 2)
                ImageCond_mask = torch.ones((ImageCond[:,:,0].shape)).to(torch.bool).to(ImageCond.device)
            
                Output.append(Output_image)
                
        
        Output = torch.cat(Output, dim=1).squeeze()
            
        return Output
    

class CausalConv3d(torch.nn.Conv3d):
    # inspired by https://github.com/pytorch/pytorch/issues/1333
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=[1,1,1],
                 dilation=[1,1,1],
                 groups=1,
                 bias=True):

        super(CausalConv3d, self).__init__(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
            dilation=dilation,
            groups=groups,
            bias=bias)
    
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
    
    def compute_padding(self, Dim_in, k_dim):
        padding = self.dilation[k_dim]*(self.kernel_size[k_dim] - 1) + 1 - self.stride[k_dim] - Dim_in*(1-self.stride[k_dim])
        
        return padding
        
    def temporal_causal_padding(self, video, padding_len, padding_values = None):
        
        """
        padding_values (B, C, D, H, W): is not none padding replicate the first element in the D dimension
        """
        
        if padding_values is None:
            padding = video[:,:,0,:,:].unsqueeze(2).expand(-1,-1,padding_len,-1,-1)
        else:
            padding = padding_values[:,:,0,:,:].unsqueeze(2).expand(-1,-1,
                                                                    padding_len-padding_values.shape[2],
                                                                    -1,-1)
            padding = torch.cat([padding,
                                 padding_values], dim = 2)
            
        #print(padding)
            
            
            
        padded_video = torch.cat([padding,
                                  video], dim = 2)
        
        return padded_video
    
    def spatial_padding(self, video, padding_len_h, padding_len_w):
        
        w_pad = np.ceil(padding_len_w/2).astype(int)
        h_pad = np.ceil(padding_len_h/2).astype(int)
        
        padding = (w_pad,w_pad,
                   h_pad,h_pad,
                   0,0)
        
        padded_video = torch.nn.functional.pad(video, pad = padding, mode='replicate')
        
        return padded_video
        
        
    def forward(self, input, conditional_padding = None):

        temporal_padding_len = self.compute_padding(input.shape[2], 0)
        #print(temporal_padding_len)
        spatial_h_padding_len = self.compute_padding(input.shape[3], 1)
        #print(spatial_h_padding_len)
        spatial_w_padding_len = self.compute_padding(input.shape[4], 2)
        #print(spatial_w_padding_len)
        
        
        time_padded_video = self.temporal_causal_padding(input, padding_len = temporal_padding_len, 
                                                         padding_values = conditional_padding)
        
        full_padded_video = self.spatial_padding(time_padded_video,
                                                 spatial_h_padding_len,
                                                 spatial_w_padding_len)
        
        return super(CausalConv3d, self).forward(full_padded_video)

class FullAttention_CausalConv(nn.Module):
    def __init__(self,
                weather_CHW_dim = [10, 9, 12],
                cb_emb_dim = 16,
                cb_heads = 4,
                channels_cb = 32,
                channels_wb = 32,
            #  convlstm_IO_units = 16,
            #  convlstm_hidden_units = 32,
            #  convlstm_nlayer = 3,
            #  convlstm_kernel = 5,
                cconv3d_input_channels = 16,
                cconv3d_hidden_channels = [32],
                cconv3d_kernel = (3,5,5),
                cconv3d_dilation = (1,1,1),
                cconv3d_layers = 1,
                densification_dropout = 0.5,
                upsampling_dim = [104, 150],
                spatial_dropout = 0.35,
                layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        
        self.cconv3d_input_channels = cconv3d_input_channels
        self.cconv3d_hidden_channels = cconv3d_hidden_channels
        self.cconv3d_kernel = cconv3d_kernel
        self.cconv3d_dilation = cconv3d_dilation
        self.cconv3d_layers = cconv3d_layers
        
        
        
        self.cb_emb_dim = cb_emb_dim
        self.cb_heads = cb_heads
        self.channels_cb = channels_cb
        self.channels_wb = channels_wb
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.cb_emb_dim,
                 heads = self.cb_heads,
                 output_channels = self.cconv3d_input_channels)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(int(self.cconv3d_input_channels*2))
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Attention_Block(
                 embedding_dim = self.cb_emb_dim,
                 input_channles = self.input_dimension[0],
                 heads = self.cb_heads,
                 output_dims = [self.cconv3d_input_channels, *self.upsampling_dim])
        
        ### Join Modoule ### 
        
        self.Joint_Conv3d = nn.Sequential(nn.Conv3d(int(self.cconv3d_input_channels*2),
                                                self.cconv3d_input_channels,
                                                kernel_size=(1,5,5), padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(self.cconv3d_input_channels, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    nn.LeakyReLU())
        
        input_features = self.cconv3d_input_channels
        
        for i in range(self.cconv3d_layers):
                
            setattr(self, f"CausalConv3d_{i}",
                    CausalConv3d(input_channels = input_features,
                                output_channels = self.cconv3d_hidden_channels[i],
                                    kernel_size=self.cconv3d_kernel[i],
                                    dilation = self.cconv3d_dilation[i]))
            
            
            setattr(self, f"LayerNorm_Act_{i}",
                    nn.Sequential(LayerNorm_MA(self.cconv3d_hidden_channels[i], move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                    nn.LeakyReLU()))
            
            input_features = self.cconv3d_hidden_channels[i]
            
        
        self.Output_layer = nn.Sequential(
                                        MoveAxis(1,-1),
                                        nn.Linear(self.cconv3d_hidden_channels[-1], self.cconv3d_input_channels),
                                        LayerNorm_MA(self.cconv3d_input_channels,
                                                     elementwise_affine = self.layernorm_affine),
                                        nn.LeakyReLU(),
                                        nn.Linear(self.cconv3d_input_channels, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
            
        #Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # alt N, D, C  # N, C, D, 2
        #date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Upsampled_VideoCond = []
            Upsampled_VideoWeather = []
            # Target_VideoCond = []
            # Upsampled_VideoWeather = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                Upsampled_VideoCond.append(self.Icondition_Module(X_t, Z, X_mask_t))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_VideoWeather.append(self.Weather_Module(Weather_Keys,
                                                                       Weather_Values,
                                                                       Z.flatten(start_dim = 1, end_dim = 2)
                                                                       )) # K, V, Q
                
            Upsampled_VideoCond = torch.stack(Upsampled_VideoCond, dim = 2)
            Upsampled_VideoWeather = torch.stack(Upsampled_VideoWeather, dim = 2)
            
            date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                self.upsampling_dim[0],
                                                                                self.upsampling_dim[1],
                                                                                -1)
            # print(Weaether_seq.shape)
            # print(Target_VideoCond.shape)
            Joint_seq = torch.cat([Upsampled_VideoWeather,
                                    Upsampled_VideoCond], 
                                    dim = 1)
            
            Joint_seq = (Joint_seq * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
            
            
            Joint_seq = self.Joint_Conv3d(Joint_seq)
            
            #Joint_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)
            
            ### Sequential module ### 
            
            for i in range(self.cconv3d_layers):
                
                Joint_seq = getattr(self, f"CausalConv3d_{i}")(Joint_seq)
                Joint_seq = getattr(self, f"LayerNorm_Act_{i}")(Joint_seq)
                
            Output_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)          
            
            Output = self.Output_layer(Output_seq)
            
            return Output.squeeze()
        
        else:
            
            ImageCond = X
            ImageCond_mask = X_mask
            Joint_seq = [[] for i in range(self.cconv3d_layers)]
            Output = []
            conditional_padding_cache = 0
            
            
            conditional_padding = [None for i in range(self.cconv3d_layers)]
            
            for timestep in range(W[0].shape[2]):
                
                Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_ImageWeather = self.Weather_Module(Weather_Keys,
                                                        Weather_Values,
                                                        Z.flatten(start_dim = 1, end_dim = 2)
                                                        )
                
                date_conditioning_wm_Image = date_conditioning_wm[:,:,timestep,:]
                
                
                date_conditioning_wm_Image = date_conditioning_wm_Image[:,:,None,None,:].expand(-1,-1,
                                                                                self.upsampling_dim[0],
                                                                                self.upsampling_dim[1],
                                                                                -1)
                
                Joint_Image = torch.cat([Upsampled_ImageWeather,
                                    Upsampled_ImageCond], 
                                    dim = 1)
                
                Joint_Image = (Joint_Image * date_conditioning_wm_Image[:,:,:,:,0]) + date_conditioning_wm_Image[:,:,:,:,1]
                Joint_Image = Joint_Image.unsqueeze(2)
                
                Joint_Image = self.Joint_Conv3d(Joint_Image)
                
                
            
                ### Sequential module ###         
                for i in range(self.cconv3d_layers):
                    #print(f"{timestep}-{i}")

                    Joint_seq[i].append(Joint_Image)
                    conditional_padding[i] = torch.cat(Joint_seq[i], dim = 2)
                    
                    Joint_Image = getattr(self, f"CausalConv3d_{i}")(Joint_Image, conditional_padding[i])
                    Joint_Image = getattr(self, f"LayerNorm_Act_{i}")(Joint_Image)
                    
                    if len(Joint_seq[i]) == getattr(self, f"CausalConv3d_{i}").compute_padding(Joint_Image.shape[2], 0):
                        #print(f"Reset-{i}")
                        Joint_seq[i] = []
                
                # Spatial Dropout
                Output_image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
                
                Output_image = self.Output_layer(Output_image)
            
                ImageCond = torch.cat([Z.clone(),
                                       Output_image[:,0,:,:,:]],
                                      dim=-1)
                ImageCond = ImageCond.flatten(start_dim = 1, end_dim = 2)
                ImageCond_mask = torch.ones((ImageCond[:,:,0].shape)).to(torch.bool).to(ImageCond.device)
            
                Output.append(Output_image)
                
        
        Output = torch.cat(Output, dim=1).squeeze()
            
        return Output


### Vision Transformer

class MHA_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 heads):
        super().__init__()
        
        
        self.norm_layer_1 = nn.LayerNorm(normalized_shape = embedding_dim)
        
        self.mha = nn.MultiheadAttention(embedding_dim, heads,
                                         batch_first=True)
        self.norm_layer_2 = nn.LayerNorm(normalized_shape = embedding_dim)
        
        
        self.mlp = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    nn.GELU(),
                                    nn.Linear(embedding_dim, embedding_dim),
                                    )
        
    def forward(self, input):
        
        skip_1 = input.clone()
        output = self.norm_layer_1(input)
        
        output, _ = self.mha(
                            query = output, #(N,L,E)
                            key = output,
                            value = output
                            )
        
        output = output + skip_1
        skip_2 = output.clone()
        
        output = self.norm_layer_2(output)
        output = self.mlp(output)
        
        output = output + skip_2
        
        return output
        

class FullAttention_ViT(nn.Module):
    
    def depatchify(self, batch, patch_size, image_size):
        """
        Patchify the batch of images
            
        Shape:
            batch: (b, nd*nh*nw, c*pd*ph*pw)
            output: (b, c, d, h, w)
        """
        b, lenght, emb_dim = batch.shape
        pd, ph, pw = patch_size
        d, h, w = image_size
        
        c = emb_dim//(pd*ph*pw)
        
        batch_patches = torch.reshape(batch, (b, c, d, h, w))

        return batch_patches
    
    def patchify(self, batch, patch_size):
        """
        Patchify the batch of images
            
        Shape:
            batch: (b, c, d, h, w)
            output: (b, nh*nw, c*ph*pw)
        """
        b, c, d, h, w = batch.shape
        pd, ph, pw = patch_size
        nd, nh, nw = d // pd, h // ph, w // pw

        batch_patches = torch.reshape(batch, (b, nd*nh*nw, c*pd*ph*pw))

        return batch_patches
    
    def PositionEmbedding(self, seq_len, emb_size):
        embeddings = torch.ones((seq_len, emb_size))
        for i in range(seq_len):
            for j in range(emb_size):
                embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size))) if j % 2 == 0 else np.cos(i / (pow(10000, (j - 1) / emb_size)))
        
        embeddings = embeddings.clone().requires_grad_()
        return embeddings
    
    def __init__(self,
                weather_CHW_dim = [7, 9, 12],
                sparse_emb_dim = 32,
                sparse_heads = 2,
                dense_emb_dim = 128,
                dense_emb_kernel = (2,2,2),
                dense_heads = 4,
                patch_size = (2,2,2),
                mha_blocks = 3,
                densification_dropout = 0.5,
                upsampling_dim = [4,42,62],
                spatial_dropout = 0.35,
                layernorm_affine = False):
        
        super().__init__()
        
        self.input_dimension = weather_CHW_dim
        self.sparse_emb_dim = sparse_emb_dim
        self.sparse_heads = sparse_heads
        self.patch_size = patch_size
        self.dense_emb_kernel = dense_emb_kernel
        self.dense_emb_dim = dense_emb_dim
        self.dense_heads = dense_heads
        self.mha_blocks = mha_blocks
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.Icondition_Module = Conditioning_Attention_Block(embedding_dim = self.sparse_emb_dim,
                 heads = self.sparse_heads,
                 output_channels = self.sparse_emb_dim)
        
        self.Date_Conditioning_Module_wm = Date_Conditioning_Block(int(self.sparse_emb_dim*2))
        
        ### Weather module ### 
        
        # TODO 
        # input_dimensions = 
        
        self.Weather_Module = Weather_Attention_Block(
                 embedding_dim = self.sparse_emb_dim,
                 input_channles = self.input_dimension[0],
                 heads = self.sparse_heads,
                 output_dims = [self.sparse_emb_dim, *self.upsampling_dim[1:]])
        
        ### Join Modoule ### 
        
        # self.dense_embedding = CausalConv3d(input_channels = int(self.sparse_emb_dim*2),
        #                         output_channels = self.dense_emb_dim,
        #                             kernel_size=self.dense_emb_kernel)
        
        #instead of patchify:
        self.dense_embedding = nn.Conv3d(int(self.sparse_emb_dim*2),
                                         self.dense_emb_dim,
                                         kernel_size = self.patch_size,
                                         stride = self.patch_size,
                                         padding="valid")
        
        ## ViT blocks ## 
        
        nd = self.upsampling_dim[0] // self.patch_size[0]
        nh = self.upsampling_dim[1] // self.patch_size[1]
        nw = self.upsampling_dim[2] // self.patch_size[2]
        
        dense_seq_len = nd*nh*nw
        
        #dense_emb_dim = int(self.sparse_emb_dim*2) #c*ph*pw
        self.pasitional_embedding = self.PositionEmbedding(dense_seq_len, self.dense_emb_dim) # (seq_len, emb_dim)
        
        
        for i in range(self.mha_blocks):
            setattr(self, f"MHA_Block_{i}",
                    MHA_Block(self.dense_emb_dim, self.dense_heads))
            
            
        # depatchify
        
        output_channels = self.dense_emb_dim//(math.prod(self.patch_size))
        
        self.linear = nn.Sequential(
                                    MoveAxis(1,-1),
                                    nn.Linear(output_channels, output_channels),
                                    nn.GELU(),
                                    nn.Linear(output_channels, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
            
        #Weaether_seq = self.Weather_Module(W[0]) # N, C, D, H, W
        date_conditioning_wm = self.Date_Conditioning_Module_wm(W[1]) # alt N, D, C  # N, C, D, 2
        #date_conditioning_sm = self.Date_Conditioning_Module_sm(W[1]) # N, C, D, 2
        
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Upsampled_VideoCond = []
            Upsampled_VideoWeather = []
            # Target_VideoCond = []
            # Upsampled_VideoWeather = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                Upsampled_VideoCond.append(self.Icondition_Module(X_t, Z, X_mask_t))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_VideoWeather.append(self.Weather_Module(Weather_Keys,
                                                                       Weather_Values,
                                                                       Z.flatten(start_dim = 1, end_dim = 2)
                                                                       )) # K, V, Q
                
            Upsampled_VideoCond = torch.stack(Upsampled_VideoCond, dim = 2)
            Upsampled_VideoWeather = torch.stack(Upsampled_VideoWeather, dim = 2)
            
            date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                self.upsampling_dim[1],
                                                                                self.upsampling_dim[2],
                                                                                -1)
            # print(Weaether_seq.shape)
            # print(Target_VideoCond.shape)
            Hidden_Video = torch.cat([Upsampled_VideoWeather,
                                    Upsampled_VideoCond], 
                                    dim = 1)
            
            Hidden_Video = (Hidden_Video * date_conditioning_wm[:,:,:,:,:,0]) + date_conditioning_wm[:,:,:,:,:,1]
            
            print(Hidden_Video.shape)
            Hidden_Video = self.dense_embedding(Hidden_Video)
            print(Hidden_Video.shape)
            Hidden_Video = torch.moveaxis(Hidden_Video, 1,-1).flatten(1,3)
            print(Hidden_Video.shape)
            
            pasitional_embedding = self.pasitional_embedding[None,:,:].expand(Hidden_Video.shape[0],
                                                                              -1,-1)  # (seq_len, emb_dim)
            
            Hidden_Video = Hidden_Video + pasitional_embedding
            
            for i in range(self.mha_blocks):
            
                Hidden_Video = getattr(self, f"MHA_Block_{i}")(Hidden_Video)
            
            Hidden_Video = self.depatchify(Hidden_Video,
                                           self.patch_size,
                                           self.upsampling_dim)
            
            print(Hidden_Video.shape)
            #Output_seq = nn.functional.dropout3d(Joint_seq, p = self.spatial_dropout, training= True)          
            
            Output_Video = self.linear(Hidden_Video)
            
            return Output_Video.squeeze()
        
        else:
            
            ImageCond = X
            ImageCond_mask = X_mask
            Hidden_Frames = []
            Hidden_Video = []
            Output_Video = []
            
            # Joint_seq[i].append(Joint_Image)
            # conditional_padding[i] = torch.cat(Joint_seq[i], dim = 2)
            
            for timestep in range(W[0].shape[2]):
                
                Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask).unsqueeze(2)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Upsampled_ImageWeather = self.Weather_Module(Weather_Keys,
                                                        Weather_Values,
                                                        Z.flatten(start_dim = 1, end_dim = 2)
                                                        ).unsqueeze(2)
                
                date_conditioning_Frames = date_conditioning_wm[:,:,:timestep+1,:] 
                date_conditioning_Frames = date_conditioning_Frames[:,:,:,None,None,:].expand(-1,-1,-1,
                                                                                self.upsampling_dim[1],
                                                                                self.upsampling_dim[2],
                                                                                -1)
                
                 # N, C, D, 2
                # date_conditioning_wm = date_conditioning_wm[:,:,:,None,None,:].expand(-1,-1,-1,
                #                                                                 self.upsampling_dim[1],
                #                                                                 self.upsampling_dim[2],
                #                                                                 -1)
                
                Hidden_Frame = torch.cat([Upsampled_ImageWeather,
                                    Upsampled_ImageCond], 
                                    dim = 1)
                
                Hidden_Frames.append(Hidden_Frame)
                Hidden_Video = torch.cat(Hidden_Frames, dim = 2)
                Hidden_Video_tlen = Hidden_Video.shape[2]
                print(Hidden_Video.shape)
                
                # Pad if len is too short
                if Hidden_Video_tlen<self.patch_size[0]:
                
                    padding_len = self.patch_size[0]-Hidden_Video_tlen
                    padding_video = Hidden_Video[:,:,0,:,:].unsqueeze(2).expand(-1,-1,
                                                                    padding_len,
                                                                    -1,-1)
                    
                    padding_date_conditioning = date_conditioning_Frames[:,:,0,:,:,:].unsqueeze(2).expand(-1,-1,
                                                                    padding_len,
                                                                    -1,-1,-1)
                    
                    date_conditioning_Frames = torch.cat([padding_date_conditioning,
                                        date_conditioning_Frames], dim = 2)
                    
                    Hidden_Video = torch.cat([padding_video,
                                        Hidden_Video], dim = 2)
                    
                    print("Padding HV", Hidden_Video.shape)
                    
                    Hidden_Video_tlen = Hidden_Video.shape[2]
                
                # Date Conditioning
                Hidden_Video = (Hidden_Video * date_conditioning_Frames[:,:,:,:,:,0]) + date_conditioning_Frames[:,:,:,:,:,1]
                
                # Tubelet Embedding
                Hidden_Video = self.dense_embedding(Hidden_Video)
                print(Hidden_Video.shape)
                Hidden_Video = torch.moveaxis(Hidden_Video, 1,-1).flatten(1,3)
                
                # Positional Embedding
                pasitional_embedding = self.PositionEmbedding(Hidden_Video.shape[1], self.dense_emb_dim)
                pasitional_embedding = pasitional_embedding[None,:,:].expand(Hidden_Video.shape[0],
                                                                              -1,-1)  # (seq_len, emb_dim)
            
                Hidden_Video = Hidden_Video + pasitional_embedding
                
                for i in range(self.mha_blocks):
            
                    Hidden_Video = getattr(self, f"MHA_Block_{i}")(Hidden_Video)
                
                print(Hidden_Video.shape)
                Hidden_Video = self.depatchify(Hidden_Video,
                                            self.patch_size,
                                            [Hidden_Video_tlen,*self.upsampling_dim[1:]]) 
                
                Output_Video = self.linear(Hidden_Video[:,:,padding_len:,:,:])
            
                # ### Sequential module ###         
                # for i in range(self.convlstm_nlayer):
                    
                #     # if timestep == 0:
                #     #     convlstm_h_state[i] = getattr(self, f"HiddenState_convLSTM_{i}")(Upsampled_ImageCond)
                #     #     convlstm_c_state[i] = convlstm_h_state[i]
                        
                    
                #     Joint_Image, (convlstm_h_state[i], convlstm_c_state[i]) = getattr(self, f"convLSTM_{i}")(Joint_Image,
                #                                                                                               convlstm_h_state[i],
                #                                                                                               convlstm_c_state[i])
                
                
                # # Spatial Dropout
                # Output_image =  nn.functional.dropout3d(Joint_Image, p = self.spatial_dropout, training = mc_dropout) 
                
                # Output_image = self.Output_layer(Output_image)
            
                ImageCond = torch.cat([Z.clone(),
                                       Output_Video[:,-1,:,:,:]],
                                      dim=-1)
                ImageCond = ImageCond.flatten(start_dim = 1, end_dim = 2)
                ImageCond_mask = torch.ones((ImageCond[:,:,0].shape)).to(torch.bool).to(ImageCond.device)
            
                #Output_Video.append(Output_Frames)
                
        
        Output_Video = Output_Video.squeeze()
            
        return Output_Video



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
    