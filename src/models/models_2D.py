from functools import partial
import copy
import numpy as np
import math
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.dataset_1d import Dataset_1D

from st_moe_pytorch import MoE
from st_moe_pytorch import SparseMoEBlock

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
                 output_dims,
                 activation,
                 elementwise_affine):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.output_dims = output_dims
        
        cb_topo_emb = []
        cb_topo_emb.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
        cb_topo_emb.append(self.activation)
        self.cb_topo_emb = nn.Sequential(*cb_topo_emb)
        
        cb_value_emb = []
        cb_value_emb.append(nn.Linear(input_channles, embedding_dim))
        cb_value_emb.append(self.activation)
        self.cb_value_emb = nn.Sequential(*cb_value_emb)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear_1 = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = int(embedding_dim*2), elementwise_affine=self.elementwise_affine),
                                    nn.Linear(int(embedding_dim*2), int(embedding_dim*2)),
                                    self.activation,
                                    nn.Linear(int(embedding_dim*2), output_dims[0]),
                                    self.activation
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
            
            weather_out = torch.cat([weather_out,
                                     queries], dim = -1)
            
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
                 output_channels,
                 activation,
                 elementwise_affine):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        cb_topo_emb = []
        cb_topo_emb.append(nn.Linear(3, embedding_dim))
        cb_topo_emb.append(self.activation)
        self.cb_topo_emb = nn.Sequential(*cb_topo_emb)
        
        cb_value_emb = []
        cb_value_emb.append(nn.Linear(4, embedding_dim))
        cb_value_emb.append(self.activation)
        self.cb_value_emb = nn.Sequential(*cb_value_emb)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear_1 = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = int(embedding_dim*2), elementwise_affine=self.elementwise_affine),
                                    nn.Linear(int(embedding_dim*2), int(embedding_dim*2)),
                                    self.activation,
                                    nn.Linear(int(embedding_dim*2), output_channels),
                                    self.activation
                                    )
        
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
            
            target_Icond = torch.cat([target_Icond,
                                      queries], dim = -1)
            
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
                 n_channels,
                 activation):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.n_channels = n_channels
        
        for i in range(n_channels):
            
            setattr(self, f"fc_0_{i}",
                    nn.Linear(2, 32))
            setattr(self, f"activation_0_{i}",
                    self.activation),
            setattr(self, f"fc_1_{i}",
                    nn.Linear(32, 2))
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
                 bias=True,
                 spatial_padding = True):

        super(CausalConv3d, self).__init__(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
            dilation=dilation,
            groups=groups,
            bias=bias)
    
        self.spatial_padding = spatial_padding
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
    
    def spatial_padding_fn(self, video, padding_len_h, padding_len_w):
        
        w_pad = padding_len_w//2 #np.ceil(padding_len_w/2).astype(int)
        w_modulo_pad = padding_len_w%2
        h_pad = padding_len_h//2 #np.ceil(padding_len_h/2).astype(int)
        h_modulo_pad = padding_len_h%2
        
        padding = (w_pad+w_modulo_pad,w_pad,
                   h_pad+h_modulo_pad,h_pad,
                   0,0)
        
        padded_video = torch.nn.functional.pad(video, pad = padding, mode='replicate')
        
        return padded_video
        
        
    def forward(self, input, conditional_padding = None):

        temporal_padding_len = self.compute_padding(input.shape[2], 0)
        #print(temporal_padding_len)
        
        padded_video = self.temporal_causal_padding(input, padding_len = temporal_padding_len, 
                                                         padding_values = conditional_padding)
        
        if self.spatial_padding is True:
            spatial_h_padding_len = self.compute_padding(input.shape[3], 1)
            #print(spatial_h_padding_len)
            spatial_w_padding_len = self.compute_padding(input.shape[4], 2)
            #print(spatial_w_padding_len)
        
            padded_video = self.spatial_padding_fn(padded_video,
                                                    spatial_h_padding_len,
                                                    spatial_w_padding_len)
            
        return super(CausalConv3d, self).forward(padded_video)
    
class CausalAvgPool3d(torch.nn.AvgPool3d):
    # inspired by https://github.com/pytorch/pytorch/issues/1333
    def __init__(self,
                 kernel_size,
                 stride=None,
                 spatial_padding = True):
        
        self.spatial_padding = spatial_padding
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

        super(CausalAvgPool3d, self).__init__(
            self.kernel_size,
            self.stride)
    
    def compute_padding(self, Dim_in, k_dim):
        padding = (self.kernel_size[k_dim] - 1) + 1 - self.stride[k_dim] - Dim_in*(1-self.stride[k_dim])
        
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
    
    def spatial_padding_fn(self, video, padding_len_h, padding_len_w):
        
        w_pad = padding_len_w//2 #np.ceil(padding_len_w/2).astype(int)
        w_modulo_pad = padding_len_w%2
        h_pad = padding_len_h//2 #np.ceil(padding_len_h/2).astype(int)
        h_modulo_pad = padding_len_h%2
        
        padding = (w_pad+w_modulo_pad,w_pad,
                   h_pad+h_modulo_pad,h_pad,
                   0,0)
        
        padded_video = torch.nn.functional.pad(video, pad = padding, mode='replicate')
        
        return padded_video
        
        
    def forward(self, input, conditional_padding = None):

        temporal_padding_len = self.compute_padding(input.shape[2], 0)
        #print(temporal_padding_len)
        
        padded_video = self.temporal_causal_padding(input, padding_len = temporal_padding_len, 
                                                         padding_values = conditional_padding)
        
        if self.spatial_padding is True:
            spatial_h_padding_len = self.compute_padding(input.shape[3], 1)
            #print(spatial_h_padding_len)
            spatial_w_padding_len = self.compute_padding(input.shape[4], 2)
            #print(spatial_w_padding_len)
        
            padded_video = self.spatial_padding_fn(padded_video,
                                                    spatial_h_padding_len,
                                                    spatial_w_padding_len)
            
        return super(CausalAvgPool3d, self).forward(padded_video)
    

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
                 heads,
                 activation,
                 dropout_p,
                 elementwise_affine):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        self.dropout_p = dropout_p
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        
        self.norm_layer_1 = nn.LayerNorm(normalized_shape = embedding_dim, 
                                         elementwise_affine = self.elementwise_affine)
        
        self.mha = nn.MultiheadAttention(embedding_dim, heads,
                                         batch_first=True)
        
        #self.dropout_1 = nn.Dropout1d(self.dropout_p)
        
        self.norm_layer_2 = nn.LayerNorm(normalized_shape = embedding_dim,
                                         elementwise_affine = self.elementwise_affine)
        
        
        self.mlp = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, embedding_dim),
                                    )
        
        #self.dropout_2 = nn.Dropout1d(self.dropout_p)
        
    def forward(self, input, attn_mask = None, mc_dropout = True):
        
        skip_1 = input #.clone()
        output = self.norm_layer_1(input)
        
        output, _ = self.mha(
                            query = output, #(N,L,E)
                            key = output,
                            value = output,
                            attn_mask = attn_mask,
                            is_causal = True
                            )
        
        #output = self.dropout_1(torch.moveaxis(output, source = 1, destination = -1))
        
        output = nn.functional.dropout1d(torch.moveaxis(output, source = 1, destination = -1),
                                         p = self.dropout_p, training = self.training or mc_dropout)
        
        output = torch.moveaxis(output, source = 1, destination = -1)
        
        output = output + skip_1
        skip_2 = output #.clone()
        
        output = self.norm_layer_2(output)
        output = self.mlp(output)
        
        #output = self.dropout_2(torch.moveaxis(output, source = 1, destination = -1))
        output = nn.functional.dropout1d(torch.moveaxis(output, source = 1, destination = -1),
                                         p = self.dropout_p, training = self.training or mc_dropout)
        
        output = torch.moveaxis(output, source = 1, destination = -1)
        
        output = output + skip_2
        
        output = self.activation(output)
        
        return output
        

class ViViT_STMoE(nn.Module):
    
    def depatchify(self, batch, patch_size, image_size):
        """
        Patchify the batch of images
            
        Shape:
            batch: (b, nd*nh*nw, c*pd*ph*pw)
            output: (b, c, d, h, w)
        """
        b, lenght, emb_dim = batch.shape
        _, ph, pw = patch_size
        d, h, w = image_size
        
        c = emb_dim//(ph*pw)
        
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
        
        #embeddings = embeddings.clone().requires_grad_()
        return embeddings
    
    def __init__(self,
                weather_CHW_dim = [23, 9, 12],
                spatial_embedding_dim = 32,
                spatial_heads = 4,
                fusion_embedding_dim = 128,
                st_heads = 4,
                patch_size = (2,2,2),
                st_mha_blocks = 3,
                num_experts = 64,
                densification_dropout = 0.5,
                upsampling_dim = [4,42,62],
                spatial_dropout = 0.0, # TODO
                layernorm_affine = False,
                activation = "LeakyReLU"):
        
        super().__init__()
        
        self.weather_dim = weather_CHW_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.patch_size = patch_size
        self.fusion_embedding_dim = fusion_embedding_dim
        self.st_heads = st_heads
        self.st_mha_blocks = st_mha_blocks
        self.num_experts = num_experts
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
            
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.SparseAutoreg_Module = Spatial_Attention_Block(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 5,
                                                            hidden_channels = 16, 
                                                            out_channels = 1, 
                                                            activation = self.activation) 
        
        # self.ST_Conditioning_CausalAvgPool = CausalAvgPool3d(kernel_size = self.patch_size,
        #                                                     stride = [1,*self.patch_size[1:]],
        #                                                     spatial_padding = False)
        
        self.ST_Conditioning_CausalConv3d = CausalConv3d(1,1,
                                                        kernel_size = self.patch_size,
                                                        stride = [1,*self.patch_size[1:]],
                                                        spatial_padding = False)
        
        self.Weather_Module = Spatial_Attention_Block(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine)
        
        ### Tublet Embedding ### 
        
        Tublet_Embedding = []
            
        if self.upsampling_dim[1]%self.patch_size[1] != 0 or self.upsampling_dim[2]%self.patch_size[2] != 0:
        
            # h padding
            pad_h = self.patch_size[1] - (self.upsampling_dim[1]%self.patch_size[1])
            
            # w padding 
            pad_w = self.patch_size[2] - (self.upsampling_dim[2]%self.patch_size[2])
            
            paw_w_r = pad_w//2 + pad_w%2
            paw_w_l = pad_w//2
            
            paw_h_r = pad_h//2 + pad_h%2
            paw_h_l = pad_h//2
            
            self.spatial_padding = (paw_w_r,paw_w_l,
                   paw_h_r,paw_h_l,
                   0,0)
        
            Tublet_Embedding.append(nn.ReplicationPad3d(padding  = self.spatial_padding))
            
            self.nd = self.upsampling_dim[0] #// self.patch_size[0]
            self.nh = (self.upsampling_dim[1]+(self.spatial_padding[2]*2)) // self.patch_size[1]
            self.nw = (self.upsampling_dim[2]+(self.spatial_padding[0]*2)) // self.patch_size[2]
            
        else:
            self.spatial_padding = None
            
            self.nd = self.upsampling_dim[0] #// self.patch_size[0]
            self.nh = self.upsampling_dim[1] // self.patch_size[1]
            self.nw = self.upsampling_dim[2] // self.patch_size[2]
            
        self.n_patches = self.nd*self.nh*self.nw
        
        Tublet_Embedding.append(CausalConv3d(int(self.spatial_embedding_dim*2),
                                         self.fusion_embedding_dim,
                                         kernel_size = self.patch_size,
                                         stride = [1,*self.patch_size[1:]],
                                         spatial_padding = False))
        Tublet_Embedding.append(LayerNorm_MA(self.fusion_embedding_dim, 1, -1))
        Tublet_Embedding.append(self.activation_fn)
        
        self.Tublet_Embedding = nn.Sequential(*Tublet_Embedding)
        
        
        ### Causal Mask 
        self.causal_mask = torch.tril(torch.ones(self.upsampling_dim[0], self.upsampling_dim[0]))
        self.causal_mask = self.causal_mask.repeat_interleave(int(self.nh*self.nw),
                                                        dim = 0)
        
        self.causal_mask = self.causal_mask.repeat_interleave(int(self.nh*self.nw),
                                                        dim = 1)
        self.causal_mask = ~self.causal_mask.to(torch.bool)
        
        for i in range(self.st_mha_blocks):
            setattr(self, f"MHA_STMoE_Block_{i}",
                    MHA_STMoE_Block(self.fusion_embedding_dim,
                                self.st_heads,
                                self.num_experts,
                                self.activation,
                                elementwise_affine = self.layernorm_affine,
                                dropout_p = self.spatial_dropout))
            
            
        # Depatchify
        
        output_channels = self.fusion_embedding_dim//(math.prod([1,*self.patch_size[1:]]))
        
        self.cconv3d = nn.Sequential(CausalConv3d(output_channels, 
                                            self.spatial_embedding_dim,
                                            kernel_size = self.patch_size,
                                            #stride = [1,*self.patch_size[1:]],
                                            spatial_padding=True),
                                        LayerNorm_MA(self.spatial_embedding_dim, 1, -1),
                                        self.activation_fn,
                                        CausalAvgPool3d(kernel_size=(2,5,5),
                                                        stride=(1,1,1),
                                                        spatial_padding=True),
                                        CausalConv3d(self.spatial_embedding_dim,
                                            self.spatial_embedding_dim,
                                            kernel_size = (5,5,5),
                                            spatial_padding=True),
                                        self.activation_fn)
        
        
        self.cconv3d_output = nn.Sequential(
                                        CausalConv3d(self.spatial_embedding_dim,
                                                int(self.spatial_embedding_dim),
                                                (5,5,5)),
                                        LayerNorm_MA(int(self.spatial_embedding_dim), 1, -1),
                                        self.activation_fn,
                                        MoveAxis(1,-1),
                                        nn.Linear(self.spatial_embedding_dim, 1))
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False,
                get_aux_loss = False):
        
        ST_conditioning_input = W[1][:,:,None,None,:].expand(-1,-1,Z.shape[1],Z.shape[2],-1)
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z[:,None,:,:,:].expand(-1,W[1].shape[1],-1,-1,-1)],
                                          dim = -1) # B, D, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) # out: N,D,H,W,1
        if self.spatial_padding is not None:
            ST_Conditionings = nn.functional.pad(torch.moveaxis(ST_Conditionings, -1, 1), pad = self.spatial_padding,
                                                 mode = "replicate")
            ST_Conditionings = self.ST_Conditioning_CausalConv3d(ST_Conditionings)
        else:
            ST_Conditionings = self.ST_Conditioning_CausalConv3d(torch.moveaxis(ST_Conditionings, -1, 1))
        
        
        if W[0].shape[2] != self.upsampling_dim[0]: 
            
            print("Computing Causal Mask...")
            
            causal_masks = torch.tril(torch.ones(W[0].shape[2], W[0].shape[2])) # Tempooral Mask
            causal_masks = causal_masks.repeat_interleave(self.nh*self.nw,  # Repeating for spatial extent
                                                        dim = 0)
            causal_masks = causal_masks.repeat_interleave(self.nh*self.nw,  # Repeating for spatial extent
                                                        dim = 1)
            
            causal_masks = ~causal_masks.to(torch.bool)
            causal_masks = causal_masks[None,:,:].expand(int(W[0].shape[0]*self.st_heads), -1,-1).to(W[0].device)
                
        else:
            causal_masks = self.causal_mask[None,:,:].expand(int(W[0].shape[0]*self.st_heads),-1,-1).to(W[0].device)
            
        ### Conditioning modules ###
        
        if self.training is True and teacher_forcing is True:
            Autoreg_st = []
            Weather_st = []
            
            for timestep in range(X.shape[1]):
                
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                    
                X_mask_t = torch.repeat_interleave(X_mask_t, self.spatial_heads, dim = 0)
                
                Autoreg_s = self.SparseAutoreg_Module(
                                                    K = X_t[:,:,:3],
                                                    V = X_t,
                                                    Q = Z.flatten(1,2),
                                                    attn_mask = ~X_mask_t[:,None,:].expand(-1,
                                                                                           int(Z.shape[1]*Z.shape[2]),
                                                                                           -1)
                                                    )
                Autoreg_s = torch.moveaxis(Autoreg_s, -1, 1)
                Autoreg_s = torch.reshape(Autoreg_s, (*Autoreg_s.shape[:2],
                                          Z.shape[1], Z.shape[2]))
                
                Autoreg_st.append(Autoreg_s)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_s = self.Weather_Module(
                                                K = Weather_Keys,
                                                V = Weather_Values,
                                                Q = Z.flatten(1,2)
                                                )
                Weather_s = torch.moveaxis(Weather_s, -1, 1)
                Weather_s = torch.reshape(Weather_s, (*Weather_s.shape[:2],
                                          Z.shape[1], Z.shape[2]))
                Weather_st.append(Weather_s)
            
            
            Autoreg_st = torch.stack(Autoreg_st, dim = 2)
            Weather_st = torch.stack(Weather_st, dim = 2)
                
            Fused_st = torch.cat([Autoreg_st,
                                    Weather_st], 
                                    dim = 1)
            
            #Fused_st += ST_Conditionings
            
            Fused_st = self.Tublet_Embedding(Fused_st)
            Fused_st += ST_Conditionings
            
            Fused_st = Fused_st.flatten(2, 4)
            
            # Fused_st =  nn.functional.dropout1d(Fused_st,
            #                                     p = self.spatial_dropout, training = True)
            Fused_st = torch.moveaxis(Fused_st, 1, -1)
            
            total_aux_loss = []
            
            for i in range(self.st_mha_blocks):
            
                Fused_st, aux_loss, _, _ = getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st, causal_masks) #TODO causal_mask
                total_aux_loss.append(aux_loss)
                
            total_aux_loss = torch.tensor(total_aux_loss).sum()
                
            if self.spatial_padding is None:
            
                Fused_st = self.depatchify(Fused_st,
                                            self.patch_size,
                                            [Autoreg_st.shape[2],
                                             *self.upsampling_dim[1:]])
                
                Fused_st = self.cconv3d(Fused_st)
                
            else:
                
                Fused_st = self.depatchify(Fused_st,
                                            self.patch_size,
                                            [Autoreg_st.shape[2],
                                            self.upsampling_dim[1] + (self.spatial_padding[2]+self.spatial_padding[3]),
                                            self.upsampling_dim[2] + (self.spatial_padding[0]+self.spatial_padding[1])])
                
                Fused_st = self.cconv3d(Fused_st)
                Fused_st = Fused_st[:,:,:,
                                self.spatial_padding[2]:-self.spatial_padding[3],
                                self.spatial_padding[0]:-self.spatial_padding[1]]
            
            
            Fused_st = Fused_st + Autoreg_st
            
            Output_st = self.cconv3d_output(Fused_st)
            
            if get_aux_loss is True:
                return [Output_st.squeeze(), total_aux_loss]
            else:
                return Output_st.squeeze()
        
        else:
            
            Sparse_data = X
            Sparse_data_mask = X_mask
            
            Autoreg_st_rlist = []
            Weather_st_rlist = []
            
            total_aux_loss = []
            
            # ImageCond = X
            # ImageCond_mask = X_mask
            # Hidden_Frames = []
            # Hidden_Video = []
            # Output_Video = []
            
            # Unfolding time - iterate own prediction as input
            for timestep in tqdm(range(W[0].shape[2])):
                
                Sparse_data_mask = torch.repeat_interleave(Sparse_data_mask, self.spatial_heads, dim = 0)
                
                Autoreg_s = self.SparseAutoreg_Module(
                                                    K = Sparse_data[:,:,:3],
                                                    V = Sparse_data,
                                                    Q = Z.flatten(1,2),
                                                    attn_mask = ~Sparse_data_mask[:,None,:].expand(-1,
                                                                                                    int(Z.shape[1]*Z.shape[2]),
                                                                                                    -1))
                Autoreg_s = torch.moveaxis(Autoreg_s, -1, 1)
                Autoreg_s = torch.reshape(Autoreg_s, (*Autoreg_s.shape[:2],
                                          Z.shape[1], Z.shape[2]))
                
                Autoreg_st_rlist.append(Autoreg_s)
                
                
                
                #Upsampled_ImageCond = self.Icondition_Module(ImageCond, Z, ImageCond_mask).unsqueeze(2)
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_s = self.Weather_Module(
                                                K = Weather_Keys,
                                                V = Weather_Values,
                                                Q = Z.flatten(1,2)
                                                )
                
                Weather_s = torch.moveaxis(Weather_s, -1, 1)
                Weather_s = torch.reshape(Weather_s, (*Weather_s.shape[:2],
                                          Z.shape[1], Z.shape[2]))
                
                Weather_st_rlist.append(Weather_s)
                
                
                
                Autoreg_st = torch.stack(Autoreg_st_rlist, dim = 2)
                Weather_st = torch.stack(Weather_st_rlist, dim = 2)
                Fused_st = torch.cat([Autoreg_st,
                                    Weather_st], 
                                    dim = 1)
            
                
                Fused_st = self.Tublet_Embedding(Fused_st)
                
                Fused_st += ST_Conditionings[:,:,:Fused_st.shape[2],:,:]
                
                Fused_st = Fused_st.flatten(2, 4)
            
                # Fused_st =  nn.functional.dropout1d(Fused_st,
                #                                     p = self.spatial_dropout, training = True)
                Fused_st = torch.moveaxis(Fused_st, 1, -1)
                
                causal_mask = causal_masks[:,
                                          :Fused_st.shape[1],
                                          :Fused_st.shape[1]]
            
                total_aux_loss = []
                
                for i in range(self.st_mha_blocks):
            
                    Fused_st, aux_loss, _, _ = getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st,
                                                                   causal_mask)
                    
                    if timestep == W[0].shape[2]-1:
                        total_aux_loss.append(aux_loss)
                
                if self.spatial_padding is None:
    
                    Fused_st = self.depatchify(Fused_st,
                                                self.patch_size,
                                                [Autoreg_st.shape[2],*self.upsampling_dim[1:]])
        
                    Fused_st = self.cconv3d(Fused_st)
        
                else:
        
                    Fused_st = self.depatchify(Fused_st,
                                                self.patch_size,
                                                [Autoreg_st.shape[2],
                                                self.upsampling_dim[1] + (self.spatial_padding[2]+self.spatial_padding[3]),
                                                self.upsampling_dim[2] + (self.spatial_padding[0]+self.spatial_padding[1])])
        
                    Fused_st = self.cconv3d(Fused_st)
                    Fused_st = Fused_st[:,:,:,
                                    self.spatial_padding[2]:-self.spatial_padding[3],
                                    self.spatial_padding[0]:-self.spatial_padding[1]]
    
    
                Fused_st = Fused_st + Autoreg_st

                Output_st = self.cconv3d_output(Fused_st)

                # Pass present prediction to the next input
                Sparse_data = torch.cat([Z.clone(),
                                    Output_st[:,-1,:,:,:]],
                                    dim=-1)
                Sparse_data = Sparse_data.flatten(start_dim = 1, end_dim = 2)
                Sparse_data_mask = torch.ones((Sparse_data[:,:,0].shape)).to(torch.bool).to(Sparse_data.device)
            
            
            total_aux_loss = torch.tensor(total_aux_loss).sum()
            # Output_Video = Output_Video.squeeze()
                
            if get_aux_loss is True:
                return [Output_st.squeeze(), total_aux_loss]
            else:
                return Output_st.squeeze()
            
            
class ViViT_STMoE_PI(ViViT_STMoE):
    
    
    def __init__(self,
                 weather_CHW_dim = [23, 9, 12],
                spatial_embedding_dim = 32,
                spatial_heads = 4,
                fusion_embedding_dim = 128,
                st_heads = 4,
                patch_size = (2,2,2),
                st_mha_blocks = 3,
                num_experts = 64,
                densification_dropout = 0.5,
                upsampling_dim = [4,42,62],
                spatial_dropout = 0.0, # TODO
                layernorm_affine = False,
                activation = "LeakyReLU"):
        
        self.weather_dim = weather_CHW_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.patch_size = patch_size
        self.fusion_embedding_dim = fusion_embedding_dim
        self.st_heads = st_heads
        self.st_mha_blocks = st_mha_blocks
        self.num_experts = num_experts
        self.densification_dropout_p = densification_dropout
        self.upsampling_dim = upsampling_dim
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation

        super(ViViT_STMoE_PI, self).__init__(
            weather_CHW_dim = self.weather_dim,
            spatial_embedding_dim = self.spatial_embedding_dim,
            spatial_heads = self.spatial_heads,
            fusion_embedding_dim = self.fusion_embedding_dim,
            st_heads = self.st_heads,
            patch_size = self.patch_size,
            st_mha_blocks = self.st_mha_blocks,
            num_experts = self.num_experts,
            densification_dropout = self.densification_dropout_p,
            upsampling_dim = self.upsampling_dim,
            spatial_dropout = self.spatial_dropout, 
            layernorm_affine = self.layernorm_affine,
            activation = self.activation)
        
        self.K_layer = nn.Sequential(
                                    MoveAxis(-1,1),
                                    nn.Conv2d(3,
                                                64,
                                                kernel_size=5, padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(64, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    self.activation_fn,
                                    nn.Conv2d(64,
                                                64,
                                                kernel_size=5, padding="same", padding_mode="replicate"),
                                    LayerNorm_MA(64, move_dim_from=1, move_dim_to=-1,
                                                 elementwise_affine = self.layernorm_affine),
                                    self.activation_fn,
                                    nn.Conv2d(64,
                                                2,
                                                kernel_size=1))
        
        self.K_lat_lon = torch.nn.Parameter(torch.normal(mean=10, std=5, size=(2,*self.upsampling_dim[1:])), requires_grad = True)
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False,
                get_aux_loss = False, K_out = False):
        
        output = super(ViViT_STMoE_PI, self).forward(X, Z, W, X_mask, teacher_forcing, mc_dropout,
                                            get_aux_loss)
        
        if K_out is True:
                
            # K_lat_lon = self.K_layer(Z)
            
            
            K_lat_lon = torch.clamp(self.K_lat_lon,
                                    min = 1e-2,
                                    max = 1e2)
            
            # K_lat_lon += (torch.ones_like(K_lat_lon)*1e2).to(K_lat_lon.device)
            
            K_lat_lon = K_lat_lon[None,:,:].expand(X.shape[0],-1,-1,-1)
        
            return [output, K_lat_lon]
                
        else:
            
            return output

###########################################
############ SparseData Models ############
###########################################
class FiLM_Conditioning_Block(nn.Module):
    """
    FiLM Conditioning Layer
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 FiLMed_layers):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.FiLMed_layers = FiLMed_layers
        
        for i in range(self.out_channels):
            
            setattr(self, f"fc_0_{i}",
                    nn.Linear(self.in_channels, self.hidden_channels))
            setattr(self, f"act_and_norm_0_{i}",
                    nn.Sequential(self.activation, nn.LayerNorm(self.hidden_channels)))
            setattr(self, f"fc_1_{i}",
                    nn.Linear(self.hidden_channels, self.hidden_channels))
            setattr(self, f"act_and_norm_1_{i}",
                    nn.Sequential(self.activation, nn.LayerNorm(self.hidden_channels))),
            setattr(self, f"fc_2_{i}",
                    nn.Linear(self.hidden_channels, int(self.FiLMed_layers*2)))
            setattr(self, f"act_2_{i}",
                    self.activation),
            
            
    def forward(self, input):
        """
        input (N, *, C)
        output (N, C, *, 2)
        """
        outputs = []
        for i in range(self.out_channels):
            output = getattr(self, f"fc_0_{i}")(input)
            output = getattr(self, f"act_and_norm_0_{i}")(output)
            output = getattr(self, f"fc_1_{i}")(output)
            output = getattr(self, f"act_and_norm_1_{i}")(output)
            output = getattr(self, f"fc_2_{i}")(output)
            output = getattr(self, f"act_2_{i}")(output)
            
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim = 1).moveaxis(1,-1)
        
        return outputs
    
class FiLM_Conditioning_Block_alt(nn.Module):
    """
    FiLM Conditioning Layer
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 FiLMed_layers):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.FiLMed_layers = FiLMed_layers
        
        for i in range(self.out_channels):
            
            setattr(self, f"fc_0_{i}",
                    nn.Linear(self.in_channels, self.hidden_channels))
            setattr(self, f"act_and_norm_0_{i}",
                    nn.Sequential(self.activation, nn.LayerNorm(self.hidden_channels)))
            setattr(self, f"fc_1_{i}",
                    nn.Linear(self.hidden_channels, self.hidden_channels))
            setattr(self, f"act_and_norm_1_{i}",
                    nn.Sequential(self.activation, nn.LayerNorm(self.hidden_channels))),
            setattr(self, f"fc_2_{i}",
                    nn.Linear(self.hidden_channels, self.FiLMed_layers))
            setattr(self, f"act_2_{i}",
                    self.activation),
            
            
    def forward(self, input):
        """
        Only Addition factor
        input (N, *, C)
        output (N, C, *, 1)
        """
        outputs = []
        for i in range(self.out_channels):
            output = getattr(self, f"fc_0_{i}")(input)
            output = getattr(self, f"act_and_norm_0_{i}")(output)
            output = getattr(self, f"fc_1_{i}")(output)
            output = getattr(self, f"act_and_norm_1_{i}")(output)
            output = getattr(self, f"fc_2_{i}")(output)
            output = getattr(self, f"act_2_{i}")(output)
            
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim = 1).moveaxis(1,-1)
        
        return outputs
    
class ST_Conditioning_Block(nn.Module):
    """
    FiLM Conditioning Layer
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 LayerNorm = True):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        layers = []
        
        layers.append(nn.Linear(self.in_channels, self.hidden_channels))
        layers.append(self.activation)
        if LayerNorm is True:
            layers.append(nn.LayerNorm(self.hidden_channels))
        layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        layers.append(self.activation)
        if LayerNorm is True:
            layers.append(nn.LayerNorm(self.hidden_channels))
        layers.append(nn.Linear(self.hidden_channels, self.out_channels))
        layers.append(self.activation)
        
        self.ST_layers = nn.Sequential(*layers)
            
    def forward(self, input):
        """
        input (B, D, S, C_in)
        output (B, D, S, C_out)
        """
        
        return self.ST_layers(input)
        
class Spatial_Attention_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 input_channles,
                 heads,
                 output_channels,
                 activation,
                 elementwise_affine):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        
        topo_embeddings = []
        topo_embeddings.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
        topo_embeddings.append(self.activation)
        self.topo_embeddings = nn.Sequential(*topo_embeddings)
        
        value_embeddings = []
        value_embeddings.append(nn.Linear(input_channles, embedding_dim))
        value_embeddings.append(self.activation)
        self.value_embeddings = nn.Sequential(*value_embeddings)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = int(embedding_dim*2), elementwise_affine=self.elementwise_affine),
                                    nn.Linear(int(embedding_dim*2), embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, output_channels),
                                    self.activation
                                    )
        
    def forward(self, K, V, Q, attn_mask = None):
            
            coords = torch.cat([K,
                                Q],
                               dim = 1)
            
            topographical_embedding = self.topo_embeddings(coords)
            
            keys = topographical_embedding[:,:K.shape[1],:]
            queries = topographical_embedding[:,K.shape[1]:,:]
            values = self.value_embeddings(V)
            
            weather_out, _ = self.cb_multihead_att_1(
                                            query = queries, #(N,L,E)
                                            key = keys,
                                            value = values,
                                            attn_mask = attn_mask #(N,L,S) True is Not Allowed
                                            )
            
            weather_out = torch.cat([weather_out,
                                     queries], dim = -1)
            
            weather_out = self.norm_linear(weather_out)

            return weather_out

class SparseData_Transformer(nn.Module):
    
    def depatchify(self, batch, patch_size, image_size):
        """
        Patchify the batch of images
            
        Shape:
            batch: (b, nd*nh*nw, c*pd*ph*pw)
            output: (b, c, d, h, w)
        """
        b, lenght, emb_dim = batch.shape
        _, ph, pw = patch_size
        d, h, w = image_size
        
        c = emb_dim//(ph*pw)
        
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
        
        #embeddings = embeddings.clone().requires_grad_()
        return embeddings
    
    def __init__(self,
                weather_CHW_dim = [7, 9, 12],
                target_dim = [14, 31],
                spatial_embedding_dim = 16,
                spatial_heads = 2,
                fusion_embedding_dim = 128,
                st_heads = 4,
                st_mha_blocks = 3,
                densification_dropout = 0.45,
                spatial_dropout = 0.2, # TODO
                layernorm_affine = True,
                activation = "GELU"):
        
        super().__init__()
        
        self.weather_dim = weather_CHW_dim
        self.target_dim = target_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.fusion_embedding_dim = fusion_embedding_dim
        self.st_heads = st_heads
        self.st_mha_blocks = st_mha_blocks
        self.densification_dropout_p = densification_dropout
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
            
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.SparseAutoreg_Module = Spatial_Attention_Block(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 5,
                                                            hidden_channels = 64, 
                                                            out_channels = 16, 
                                                            activation = self.activation)
        
        # self.ST_Conditioning_Module = FiLM_Conditioning_Block(in_channels = 5,
        #                                                     hidden_channles = 64, 
        #                                                     out_channels = self.fusion_embedding_dim, 
        #                                                     activation = self.activation)
        

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine)
        
        ### Joint Modoule ### 
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(int(self.spatial_embedding_dim*2)+16,
                                                       self.fusion_embedding_dim),
                                             self.activation_fn)
        
        ### Flatten Space-Time Dimension
        
        ### Spatio-Temporal Positional Embedding        
        # self.positional_embedding = self.PositionEmbedding(math.prod(self.target_dim),
        #                                                    self.fusion_embedding_dim)
        
        ### Causal Mask 
        self.causal_mask = torch.tril(torch.ones((self.target_dim[0], self.target_dim[0]))) # Tempooral Mask
        self.causal_mask = self.causal_mask.repeat_interleave(self.target_dim[1], # Repeating for spatial extent
                                                        dim = 0)
        self.causal_mask = self.causal_mask.repeat_interleave(self.target_dim[1], # Repeating for spatial extent
                                                        dim = 1)
        self.causal_mask = ~self.causal_mask.to(torch.bool)
        
        for i in range(self.st_mha_blocks):
            setattr(self, f"MHA_Block_{i}",
                    MHA_Block(self.fusion_embedding_dim, self.st_heads,
                              self.activation,
                              elementwise_affine = self.layernorm_affine,
                              dropout_p = self.spatial_dropout))
            
            
        ### De-Flatten Space-Time Dimension
        
        self.linear = nn.Linear(self.fusion_embedding_dim,
                                  int(self.spatial_embedding_dim*2))
                                   
        self.output = nn.Sequential(self.activation_fn,
                                    nn.Linear(int(self.spatial_embedding_dim*2),
                                                self.spatial_embedding_dim),
                                    nn.LayerNorm(self.spatial_embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.spatial_embedding_dim, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
        
        ST_conditioning_input = W[1][:,:,None,:].expand(-1,-1,Z.shape[1],-1)
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z[:,None,:,:].expand(-1,W[1].shape[1],-1,-1)],
                                          dim = -1) # B, D, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) # N,D,S,2*FiLMed,C # ST N,D,S,C_out
        
        if [W[0].shape[2], Z.shape[1]] != self.target_dim:
                
                print("Computing Causal Mask...")
            
                # Causal Mask
                causal_masks = torch.tril(torch.ones(W[0].shape[2], W[0].shape[2])) # Tempooral Mask
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                            dim = 0)
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                        dim = 1)
                
                causal_masks = ~causal_masks.to(torch.bool)
                causal_masks = causal_masks[None,:,:].expand(int(W[0].shape[0]*self.st_heads), -1,-1).to(W[0].device)

                
        else:
                causal_masks = self.causal_mask[None,:,:].expand(int(W[0].shape[0]*self.st_heads),-1,-1).to(W[0].device)
        
        if self.training is True and teacher_forcing is True:
            Autoreg_st = []
            Weather_st = []
            
            for timestep in range(X.shape[1]):
                 
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                
                X_mask_t = torch.repeat_interleave(X_mask_t, self.spatial_heads, dim = 0)
                
                Autoreg_st.append(self.SparseAutoreg_Module(
                                                                    K = X_t[:,:,:3],
                                                                    V = X_t,
                                                                    Q = Z,
                                                                    attn_mask = ~X_mask_t[:,None,:].expand(-1,Z.shape[1],-1)
                                                                    ))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_st.append(self.Weather_Module(
                                                                K = Weather_Keys,
                                                                V = Weather_Values,
                                                                Q = Z
                                                                ))
                
            Autoreg_st = torch.stack(Autoreg_st, dim = 1)
            Weather_st = torch.stack(Weather_st, dim = 1)

            Fused_st = torch.cat([Autoreg_st,
                                    Weather_st,
                                    ST_Conditionings], 
                                    dim = -1)
            
            Fused_st_extent = Fused_st.shape[1:3]
            Fused_st = torch.flatten(Fused_st, 1, 2)
            
            Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
                                                p = self.spatial_dropout, training = True)
            
            #print(Hidden_Video.shape)
            Fused_st = self.Fusion_Embedding(torch.moveaxis(Fused_st, 1, -1))
            
            
            for i in range(self.st_mha_blocks):
            
                Fused_st = getattr(self, f"MHA_Block_{i}")(Fused_st, causal_masks, self.training)
            
            #Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*self.target_dim,Fused_st.shape[-1])) 
            Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1])) 
            
            Fused_st = self.linear(Fused_st)
            
            #Fused_st = (Fused_st * ST_Conditionings[:,:,:,2,:]) + ST_Conditionings[:,:,:,3,:]
            
            Output_st = self.output(Fused_st)
            
            return Output_st.squeeze()
        
        else:
            
            Sparse_data = X
            Sparse_data_mask = X_mask
            Fused_st_list = []
                
            # Unfolding time - iterate own prediction as input
            for timestep in tqdm(range(W[0].shape[2])):
                
                Sparse_data_mask = torch.repeat_interleave(Sparse_data_mask, self.spatial_heads, dim = 0)
                
                Autoreg_s = self.SparseAutoreg_Module(K = Sparse_data[:,:,:3],
                                                                V = Sparse_data,
                                                                Q = Z,
                                                                attn_mask = ~Sparse_data_mask[:,None,:].expand(-1,Z.shape[1],-1)).unsqueeze(1)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_s = self.Weather_Module(
                                                        K = Weather_Keys,
                                                        V = Weather_Values,
                                                        Q = Z
                                                        ).unsqueeze(1)
                
                ST_conditionings_rolling = ST_Conditionings[:,timestep,:,:].unsqueeze(1) # N,D,S,2*FiLMed,C
                Fused_s = torch.cat([Autoreg_s,
                                        Weather_s,
                                        ST_conditionings_rolling], 
                                        dim = -1)
                
                Fused_st_list.append(Fused_s)
                Fused_st = torch.cat(Fused_st_list, dim = 1)
                
                Fused_st_extent = Fused_st.shape[1:3]
                Fused_st = torch.flatten(Fused_st, 1, 2)
                
                Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
                                                p = self.spatial_dropout, training = mc_dropout)
                
                Fused_st = self.Fusion_Embedding(torch.moveaxis(Fused_st, 1, -1))
                                               
                causal_mask = causal_masks[:,
                                          :math.prod(Fused_st_extent),
                                          :math.prod(Fused_st_extent)]
                
                for i in range(self.st_mha_blocks):
            
                    Fused_st = getattr(self, f"MHA_Block_{i}")(Fused_st,
                                                                   causal_mask,
                                                                   mc_dropout)
                
                
                Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1]))        
                
                
                Fused_st = self.linear(Fused_st)
            
            
                Output_st = self.output(Fused_st)
            
                Sparse_data = torch.cat([Z.clone(),
                                       Output_st[:,-1,:,:]],
                                      dim=-1)

                Sparse_data_mask = torch.ones((Sparse_data[:,:,0].shape)).to(torch.bool).to(Sparse_data.device)
            
        Output_st = Output_st.squeeze()
            
        return Output_st

#### ST MoE ####

class MHA_STMoE_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 heads,
                 num_experts,
                 activation,
                 dropout_p,
                 elementwise_affine,
                 STMoE_prenorm = True):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        self.dropout_p = dropout_p
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        
        
        # self.norm_layer_1 = nn.LayerNorm(normalized_shape = embedding_dim, 
        #                                  elementwise_affine = self.elementwise_affine)
        
        self.mha = nn.MultiheadAttention(embedding_dim, heads,
                                         batch_first=True)
        
        
        moe = MoE(
                    dim = embedding_dim,
                    num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
                    gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
                    threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
                    threshold_eval = 0.2,
                    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                    balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
                    router_z_loss_coef = 1e-3,      # loss weight for router z-loss
                )
        
        self.moe_block = SparseMoEBlock(
                            moe,
                            add_ff_before = STMoE_prenorm,
                            add_ff_after = STMoE_prenorm
                        )
        
    def forward(self, input, attn_mask = None, mc_dropout = False):
        
        skip_1 = input #.clone()
        #output = self.norm_layer_1(input)
        
        output, _ = self.mha(
                            query = input, #(N,L,E)
                            key = input,
                            value = input,
                            attn_mask = attn_mask,
                            is_causal = True if attn_mask is not None else False
                            )
        
        output = output + skip_1
        output, total_aux_loss, balance_loss, router_z_loss = self.moe_block(output)
        
        output = nn.functional.dropout1d(torch.moveaxis(output, 1, -1),
                                        p = self.dropout_p, training = mc_dropout)    
        output = torch.moveaxis(output, source = -1, destination = 1)
        
        return output, total_aux_loss, balance_loss, router_z_loss

class SparseData_STMoE(nn.Module):
    
    def depatchify(self, batch, patch_size, image_size):
        """
        Patchify the batch of images
            
        Shape:
            batch: (b, nd*nh*nw, c*pd*ph*pw)
            output: (b, c, d, h, w)
        """
        b, lenght, emb_dim = batch.shape
        _, ph, pw = patch_size
        d, h, w = image_size
        
        c = emb_dim//(ph*pw)
        
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
        
        #embeddings = embeddings.clone().requires_grad_()
        return embeddings
    
    def __init__(self,
                weather_CHW_dim = [23, 9, 12],
                target_dim = [14, 31],
                spatial_embedding_dim = 16,
                spatial_heads = 2,
                fusion_embedding_dim = 128,
                st_heads = 4,
                st_mha_blocks = 3,
                num_experts = 16,
                densification_dropout = 0.45,
                spatial_dropout = 0.2, # TODO
                layernorm_affine = True,
                activation = "GELU"):
        
        super().__init__()
        
        self.weather_dim = weather_CHW_dim
        self.target_dim = target_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.fusion_embedding_dim = fusion_embedding_dim
        self.st_heads = st_heads
        self.st_mha_blocks = st_mha_blocks
        self.num_experts = num_experts
        self.densification_dropout_p = densification_dropout
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
            
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.SparseAutoreg_Module = Spatial_Attention_Block(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 5,
                                                            hidden_channels = 16, 
                                                            out_channels = 1, 
                                                            activation = self.activation) 

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine)
        
        ### Joint Modoule ### 
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(int(self.spatial_embedding_dim*2),
                                                       self.fusion_embedding_dim),
                                             self.activation_fn)
        
        ### Flatten Space-Time Dimension
        
        ### Causal Mask 
        self.causal_mask = torch.tril(torch.ones((self.target_dim[0], self.target_dim[0]))) # Tempooral Mask
        self.causal_mask = self.causal_mask.repeat_interleave(self.target_dim[1], # Repeating for spatial extent
                                                        dim = 0)
        self.causal_mask = self.causal_mask.repeat_interleave(self.target_dim[1], # Repeating for spatial extent
                                                        dim = 1)
        self.causal_mask = ~self.causal_mask.to(torch.bool)
        
        for i in range(self.st_mha_blocks):
            setattr(self, f"MHA_STMoE_Block_{i}",
                    MHA_STMoE_Block(self.fusion_embedding_dim,
                                self.st_heads,
                                self.num_experts,
                                self.activation,
                                elementwise_affine = self.layernorm_affine,
                                dropout_p = self.spatial_dropout))
            
            
        ### De-Flatten Space-Time Dimension
        
        self.linear = nn.Sequential(
                                    nn.Linear(self.fusion_embedding_dim,
                                                self.spatial_embedding_dim),
                                    self.activation_fn)
                                   
        self.output = nn.Sequential(nn.Linear(int(self.spatial_embedding_dim),
                                                self.spatial_embedding_dim),
                                    nn.LayerNorm(self.spatial_embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.spatial_embedding_dim, 1))
        
        
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False,
                get_aux_loss = False):
        
        ST_conditioning_input = W[1][:,:,None,:].expand(-1,-1,Z.shape[1],-1)
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z[:,None,:,:].expand(-1,W[1].shape[1],-1,-1)],
                                          dim = -1) # B, D, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) # N,D,S,2*FiLMed,C # ST N,D,S,C_out
        ST_Conditionings = torch.flatten(ST_Conditionings, 1, 2)
        
        if [W[0].shape[2], Z.shape[1]] != self.target_dim:
                
                print("Computing Causal Mask...")
            
                # Causal Mask
                causal_masks = torch.tril(torch.ones(W[0].shape[2], W[0].shape[2])) # Tempooral Mask
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                            dim = 0)
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                        dim = 1)
                
                causal_masks = ~causal_masks.to(torch.bool)
                causal_masks = causal_masks[None,:,:].expand(int(W[0].shape[0]*self.st_heads), -1,-1).to(W[0].device)

                
        else:
                causal_masks = self.causal_mask[None,:,:].expand(int(W[0].shape[0]*self.st_heads),-1,-1).to(W[0].device)
        
        if self.training is True and teacher_forcing is True:
            Autoreg_st = []
            Weather_st = []
            
            for timestep in range(X.shape[1]):
                 
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                
                X_mask_t = torch.repeat_interleave(X_mask_t, self.spatial_heads, dim = 0)
                
                Autoreg_st.append(self.SparseAutoreg_Module(
                                                            K = X_t[:,:,:3],
                                                            V = X_t,
                                                            Q = Z,
                                                            attn_mask = ~X_mask_t[:,None,:].expand(-1,Z.shape[1],-1)
                                                            ))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_st.append(self.Weather_Module(
                                                    K = Weather_Keys,
                                                    V = Weather_Values,
                                                    Q = Z
                                                    ))
                
            Autoreg_st = torch.stack(Autoreg_st, dim = 1)
            Weather_st = torch.stack(Weather_st, dim = 1)

            Fused_st = torch.cat([Autoreg_st,
                                    Weather_st], 
                                    dim = -1)
            
            Fused_st_extent = Fused_st.shape[1:3]
            Fused_st = torch.flatten(Fused_st, 1, 2)
            
            # Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
            #                                     p = self.spatial_dropout, training = True)
            
            Fused_st = self.Fusion_Embedding(Fused_st)
            
            Fused_st += ST_Conditionings
            
            
            
            total_aux_loss = []
            
            for i in range(self.st_mha_blocks):
            
                Fused_st, aux_loss, _, _= getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st, causal_masks,
                                                                                mc_dropout = True)
                total_aux_loss.append(aux_loss)
                #output, total_aux_loss, balance_loss, router_z_loss
            
            total_aux_loss = torch.tensor(total_aux_loss).sum()
            
            Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1]))
            
            Fused_st = self.linear(Fused_st)
            
            Fused_st = Fused_st + Autoreg_st
             
            Output_st = self.output(Fused_st)
            
            if get_aux_loss is True:
                return [Output_st.squeeze(), total_aux_loss]
            else:
                return Output_st.squeeze()
        
        else:
            
            Sparse_data = X
            Sparse_data_mask = X_mask
            
            Autoreg_st_rlist = []
            Weather_st_rlist = []
            
            total_aux_loss = []
                
            # Unfolding time - iterate own prediction as input
            for timestep in tqdm(range(W[0].shape[2])):
                
                Sparse_data_mask = torch.repeat_interleave(Sparse_data_mask, self.spatial_heads, dim = 0)
                
                Autoreg_st_rlist.append(self.SparseAutoreg_Module(K = Sparse_data[:,:,:3],
                                                                V = Sparse_data,
                                                                Q = Z,
                                                                attn_mask = ~Sparse_data_mask[:,None,:].expand(-1,Z.shape[1],-1))
                                )
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_st_rlist.append(self.Weather_Module(
                                                        K = Weather_Keys,
                                                        V = Weather_Values,
                                                        Q = Z
                                                        )
                                )
                
                #ST_conditionings_rolling = ST_Conditionings[:,timestep,:,:].unsqueeze(1) # N,D,S,2*FiLMed,C
                #Autoreg_st.append(Autoreg_s)
                Autoreg_st = torch.stack(Autoreg_st_rlist, dim = 1)
                Weather_st = torch.stack(Weather_st_rlist, dim = 1)
                Fused_st = torch.cat([Autoreg_st,
                                    Weather_st], 
                                    dim = -1)
                
                
                Fused_st_extent = Fused_st.shape[1:3]
                Fused_st = torch.flatten(Fused_st, 1, 2)
                
                # Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
                #                                 p = self.spatial_dropout, training = mc_dropout)
                
                
                Fused_st = self.Fusion_Embedding(Fused_st)
                Fused_st = Fused_st + ST_Conditionings[:,:Fused_st.shape[1],:]
                                               
                causal_mask = causal_masks[:,
                                          :math.prod(Fused_st_extent),
                                          :math.prod(Fused_st_extent)]
                
                for i in range(self.st_mha_blocks):
            
                    Fused_st, aux_loss, _, _ = getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st,
                                                                   causal_mask, mc_dropout = mc_dropout)
                    
                    if timestep == W[0].shape[2]-1:
                        total_aux_loss.append(aux_loss)
                
                
                Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1]))        
                
                Fused_st = self.linear(Fused_st)
            
                Fused_st = Fused_st + Autoreg_st
                
                Output_st = self.output(Fused_st)
                
                Sparse_data = torch.cat([Z.clone(),
                                       Output_st[:,-1,:,:]],
                                      dim=-1)
                
                Sparse_data_mask = torch.ones((Sparse_data[:,:,0].shape)).to(torch.bool).to(Sparse_data.device)
            
            total_aux_loss = torch.tensor(total_aux_loss).sum()
            
            if get_aux_loss is True:
                    return [Output_st.squeeze(), total_aux_loss]
            else:
                return Output_st.squeeze()
###########################################
    
class Spatial_Attention_Block_MoE(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 input_channles,
                 heads,
                 num_experts,
                 output_channels,
                 activation,
                 elementwise_affine,
                 STMoE_prenorm = True):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        self.num_experts = num_experts
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        
        topo_embeddings = []
        topo_embeddings.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
        topo_embeddings.append(self.activation)
        self.topo_embeddings = nn.Sequential(*topo_embeddings)
        
        value_embeddings = []
        value_embeddings.append(nn.Linear(input_channles, embedding_dim))
        value_embeddings.append(self.activation)
        self.value_embeddings = nn.Sequential(*value_embeddings)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        moe = MoE(
                    dim = int(embedding_dim*2),
                    num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
                    gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
                    threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
                    threshold_eval = 0.2,
                    capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                    capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                    balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
                    router_z_loss_coef = 1e-3,      # loss weight for router z-loss
                )
        
        self.moe_block = SparseMoEBlock(
                            moe,
                            add_ff_before = STMoE_prenorm,
                            add_ff_after = STMoE_prenorm
                        )
        
        self.linear = nn.Sequential(
                                    nn.Linear(int(embedding_dim*2), output_channels),
                                    self.activation,
                                    )
        
    def forward(self, K, V, Q, attn_mask = None):
            
            coords = torch.cat([K,
                                Q],
                               dim = 1)
            
            topographical_embedding = self.topo_embeddings(coords)
            
            keys = topographical_embedding[:,:K.shape[1],:]
            queries = topographical_embedding[:,K.shape[1]:,:]
            values = self.value_embeddings(V)
            
            output, _ = self.cb_multihead_att_1(
                                            query = queries, #(N,L,E)
                                            key = keys,
                                            value = values,
                                            attn_mask = attn_mask #(N,L,S) True is Not Allowed
                                            )
            
            output = torch.cat([output,
                                queries], dim = -1)
            
            output, total_aux_loss, balance_loss, router_z_loss = self.moe_block(output)
            
            output = self.linear(output)
        
        # output = nn.functional.dropout1d(torch.moveaxis(output, 1, -1),
        #                                 p = self.dropout_p, training = mc_dropout)    
        # output = torch.moveaxis(output, source = -1, destination = 1)
        
            return output, total_aux_loss, balance_loss, router_z_loss

class Spatial_STMoE(nn.Module):
    
    def __init__(self,
                weather_CHW_dim = [23, 9, 12],
                target_dim = [14, 31],
                spatial_embedding_dim = 16,
                spatial_heads = 2,
                fusion_embedding_dim = 32,
                fusion_heads = 2,
                num_experts = 16,
                densification_dropout = 0.45,
                spatial_dropout = 0.2,
                layernorm_affine = True,
                activation = "GELU"):
        
        super().__init__()
        
        self.weather_dim = weather_CHW_dim
        self.target_dim = target_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.fusion_embedding_dim = fusion_embedding_dim
        self.fusion_heads = fusion_heads
        self.num_experts = num_experts
        self.densification_dropout_p = densification_dropout
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
            
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.SparseAutoreg_Module = Spatial_Attention_Block_MoE(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                num_experts= self.num_experts,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine,
                                STMoE_prenorm = False)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 5,
                                                            hidden_channels = 32, 
                                                            out_channels = 1, 
                                                            activation = self.activation,
                                                            LayerNorm = False) 

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block_MoE(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 num_experts= self.num_experts,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine,
                 STMoE_prenorm = False)
        
        ### Joint Modoule ### 
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(int(self.spatial_embedding_dim*2),
                                                       self.fusion_embedding_dim),
                                            #nn.LayerNorm(self.fusion_embedding_dim),
                                            self.activation_fn)
        
        self.Fusion_Module = MHA_STMoE_Block(self.fusion_embedding_dim,
                                self.fusion_heads,
                                self.num_experts,
                                self.activation,
                                elementwise_affine = self.layernorm_affine,
                                dropout_p = self.spatial_dropout,
                                STMoE_prenorm = False)
        
                                   
        self.output = nn.Sequential(nn.Linear(self.fusion_embedding_dim, spatial_embedding_dim),
                                    #nn.LayerNorm(self.spatial_embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.spatial_embedding_dim, 1))
        
        
    def forward(self, X, Z, W, X_mask, mc_dropout = False,
                get_aux_loss = False):
        
        ST_conditioning_input = W[1][:,None,:].expand(-1,Z.shape[1],-1)
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z], #.expand(-1,W[1].shape[1],-1,-1)]
                                          dim = -1) # B, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) 
        #ST_Conditionings = torch.flatten(ST_Conditionings, 1, 2)
        
        if self.densification_dropout_p>0:
            GW_Values, GW_mask = densification_dropout([X,
                                                X_mask],
                                                p = self.densification_dropout_p)
        else:
            GW_Values = X
            GW_mask = X_mask
                
        GW_mask = torch.repeat_interleave(GW_mask, self.spatial_heads, dim = 0)
        
        GW_out, GW_aux_loss, _, _ = self.SparseAutoreg_Module(
                                        K = GW_Values[:,:,:3],
                                        V = GW_Values,
                                        Q = Z,
                                        attn_mask = ~GW_mask[:,None,:].expand(-1,Z.shape[1],-1)
                                        )
        
        Weather_Keys = torch.moveaxis(W[0][:,:3,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
        Weather_Values = torch.moveaxis(W[0][:,:,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
        Weather_out, Weather_aux_loss, _, _ = self.Weather_Module(
                                        K = Weather_Keys,
                                        V = Weather_Values,
                                        Q = Z
                                        )

        GW_Weather_Fusion = torch.cat([GW_out,
                                    Weather_out], 
                                    dim = -1)
            
        #Fused_st_extent = Fused_st.shape[1:3]
        #Fused_st = torch.flatten(Fused_st, 1, 2)
            
        GW_Weather_Fusion = self.Fusion_Embedding(GW_Weather_Fusion)
            
        GW_Weather_Fusion += ST_Conditionings
        
        GW_Weather_Fusion, GW_Weather_aux_loss, _, _ = self.Fusion_Module(GW_Weather_Fusion,
                                                                          mc_dropout = self.training or mc_dropout)
            
        total_aux_loss = torch.tensor([GW_Weather_aux_loss,
                                       Weather_aux_loss,
                                       GW_aux_loss]).sum()
            
        Output = self.output(GW_Weather_Fusion)
            
        #Fused_st = Fused_st + Autoreg_st
             
        if get_aux_loss is True:
            return [Output.squeeze(), total_aux_loss]
        else:
            return Output.squeeze()
    

class Spatial_STMoE_Light(nn.Module):
    
    def __init__(self,
                weather_CHW_dim = [23, 9, 12],
                target_dim = [14, 31],
                spatial_embedding_dim = 16,
                spatial_heads = 2,
                num_experts = 16,
                densification_dropout = 0.45,
                spatial_dropout = 0.2,
                layernorm_affine = True,
                activation = "GELU"):
        
        super().__init__()
        
        self.weather_dim = weather_CHW_dim
        self.target_dim = target_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_heads = spatial_heads
        self.num_experts = num_experts
        self.densification_dropout_p = densification_dropout
        self.layernorm_affine = layernorm_affine
        self.spatial_dropout = spatial_dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
            
        
        ### Conditioning module - Transofrmer like architecture ###
        
        self.SparseAutoreg_Module = Spatial_Attention_Block_MoE(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                num_experts= self.num_experts,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine)
        
        # self.ST_Conditioning_Module = ST_Conditioning_Block(
        #                                                     in_channels = 5,
        #                                                     hidden_channels = 32, 
        #                                                     out_channels = 1, 
        #                                                     activation = self.activation) 

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block_MoE(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 num_experts= self.num_experts,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine)
        
        ### Joint Modoule ### 
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(self.spatial_embedding_dim,
                                                       self.spatial_embedding_dim),
                                            #nn.LayerNorm(self.spatial_embedding_dim),
                                            self.activation_fn)
                                   
        self.output = nn.Sequential(nn.Linear(self.spatial_embedding_dim, 1))
        
        
    def forward(self, X, Z, W, X_mask, mc_dropout = False,
                get_aux_loss = False):
        
        # ST_conditioning_input = W[1][:,None,:].expand(-1,Z.shape[1],-1)
        # ST_conditioning_input = torch.cat([ST_conditioning_input,
        #                                    Z], #.expand(-1,W[1].shape[1],-1,-1)]
        #                                   dim = -1) # B, S, C
        
        # ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) 
        # #ST_Conditionings = torch.flatten(ST_Conditionings, 1, 2)
        
        if self.densification_dropout_p>0:
            GW_Values, GW_mask = densification_dropout([X,
                                                X_mask],
                                                p = self.densification_dropout_p)
        else:
            GW_Values = X
            GW_mask = X_mask
                
        GW_mask = torch.repeat_interleave(GW_mask, self.spatial_heads, dim = 0)
        
        GW_out, GW_aux_loss, _, _ = self.SparseAutoreg_Module(
                                        K = GW_Values[:,:,:3],
                                        V = GW_Values,
                                        Q = Z,
                                        attn_mask = ~GW_mask[:,None,:].expand(-1,Z.shape[1],-1)
                                        )
        
        Weather_Keys = torch.moveaxis(W[0][:,:3,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
        Weather_Values = torch.moveaxis(W[0][:,:,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
        Weather_out, Weather_aux_loss, _, _ = self.Weather_Module(
                                        K = Weather_Keys,
                                        V = Weather_Values,
                                        Q = Z
                                        )
        
        GW_Weather_Fusion = GW_out + Weather_out #+ ST_Conditionings  
            
        GW_Weather_Fusion = self.Fusion_Embedding(GW_Weather_Fusion)
            
        total_aux_loss = torch.tensor([Weather_aux_loss,
                                       GW_aux_loss]).sum()
            
        Output = self.output(GW_Weather_Fusion)
            
        #Fused_st = Fused_st + Autoreg_st
             
        if get_aux_loss is True:
            return [Output.squeeze(), total_aux_loss]
        else:
            return Output.squeeze()



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
    