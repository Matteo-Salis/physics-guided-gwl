from functools import partial
import copy

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

class Conditioning_Attention_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 heads,
                 hidden_channels,
                 output_channels,
                 padding_mode = "replicate",
                 layernorm_affine = False,
                 ):
        super().__init__()
        
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
        
        self.cb_affine_and_norm = nn.Sequential(
                                    
                                    #nn.LayerNorm(normalized_shape = embedding_dim),
                                    nn.Linear(embedding_dim, output_channels),
                                    #nn.LayerNorm(normalized_shape = embedding_dim),
                                    nn.LeakyReLU(),
                                    #nn.Dropout(p=0.25),
                                    )
        
        # self.skip_affine = nn.Linear(embedding_dim, hidden_channels)
        
        # self.cb_conv = nn.Sequential(
        #                             nn.Conv2d(embedding_dim,
        #                                        hidden_channels,
        #                                        5,
        #                                        padding='same',
        #                                        padding_mode = padding_mode),
        #                             LayerNorm_MA(hidden_channels, move_dim_from=1, move_dim_to=-1, elementwise_affine = layernorm_affine),
        #                             nn.LeakyReLU(),
        #                             nn.Conv2d(hidden_channels,
        #                                        hidden_channels,
        #                                        5,
        #                                        padding='same'),
        #                             )
        
        # self.output = nn.Sequential(LayerNorm_MA(hidden_channels, move_dim_from=1, move_dim_to=-1, elementwise_affine = layernorm_affine),
        #                             nn.LeakyReLU(),
        #                             nn.Conv2d(hidden_channels,
        #                                        output_channels,
        #                                        5,
        #                                        padding='same',
        #                                        padding_mode = padding_mode),
        #                             nn.LeakyReLU())
        
    def forward(self, X, Z, X_mask):
            
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
            
            target_Icond = self.cb_affine_and_norm(target_Icond)
            
            target_Icond = torch.moveaxis(target_Icond, -1, 1)
            target_Icond = torch.reshape(target_Icond, (*target_Icond.shape[:2],
                                         Z.shape[1], Z.shape[2]))
            
            # target_Icond_skip = self.skip_affine(torch.moveaxis(target_Icond, 1, -1))
            
            # target_Icond = self.cb_conv(target_Icond)
            
            # target_Icond = target_Icond + torch.moveaxis(target_Icond_skip, -1, 1)
            
            # target_Icond = self.output(target_Icond)

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
                    nn.LeakyReLU()),
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
    