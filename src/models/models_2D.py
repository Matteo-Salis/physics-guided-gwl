from functools import partial

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.dataset_1d import Dataset_1D

##### Blocks ######

class MoveAxis(nn.Module):
    
    def __init__(self, source, destination):
        super().__init__()
        
        self.moveaxis = partial(torch.moveaxis, source = source, destination = destination)
    
    def forward(self, x):
        return self.moveaxis(x)

class LayerNorm_MA(torch.nn.LayerNorm):
    
    def __init__(self, normalized_shape, move_dim_in = None, move_dim_out = None, eps = 0.00001, elementwise_affine = True, bias = True, device=None, dtype=None):
        super(LayerNorm_MA, self).__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        
        self.norm = []
        self.norm.append(nn.LayerNorm(normalized_shape, elementwise_affine=False))
        
        if move_dim_in is not None:
            self.norm.insert(0, MoveAxis(move_dim_in, move_dim_out)) #partial(torch.moveaxis, source = norm_dim, destination = -1)
            self.norm.append(MoveAxis(move_dim_out, move_dim_in)) #partial(torch.moveaxis, source = -1, destination = norm_dim)
            
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
    
class Weather_Upsampling_Block(nn.Module):
    
    def compute_3dTrConv_out_dim(self, h_in, k_size):
        h_out = (h_in - 1) + (k_size - 1) + 1
        return h_out
    
    def __init__(self,
                 input_dimensions = [10, 9, 12],
                 hidden_channels = 32,
                 output_channels = 16,
                 output_dim = [104, 150]):
        super().__init__()
        
        self.layers = []
        self.layers.append(nn.Conv3d(input_dimensions[0], hidden_channels, (1,5,5), padding='same', dtype=torch.float32))
        self.layers.append(LayerNorm_MA([hidden_channels,
                                        input_dimensions[1],
                                        input_dimensions[2]], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.ConvTranspose3d(hidden_channels, int(hidden_channels), (1,3,3), stride=(1,1,1), dtype=torch.float32))
        
        self.layers.append(nn.AdaptiveAvgPool3d((None,int(output_dim[0]/4),int(output_dim[1]/4)))) #padding='same'
        
        self.layers.append(LayerNorm_MA([hidden_channels, 
                                         int(output_dim[0]/4),
                                         int(output_dim[1]/4)], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.ConvTranspose3d(int(hidden_channels), int(hidden_channels), (1,3,3), stride=(1,1,1), dtype=torch.float32))
        
        # new_H = self.compute_3dTrConv_out_dim(new_H, 3)
        # new_W = self.compute_3dTrConv_out_dim(new_W, 3)
        
        # self.layers.append(LayerNorm_MA([int(hidden_channels/2),
        #                                  new_H, new_W], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.AdaptiveAvgPool3d((None,int(output_dim[0]/2),int(output_dim[1]/2)))) #padding='same'
        
        self.layers.append(nn.ConvTranspose3d(int(hidden_channels), int(hidden_channels), (1,5,5), stride=(1,1,1), dtype=torch.float32))
        
        new_H = self.compute_3dTrConv_out_dim(int(output_dim[0]/2), 5)
        new_W = self.compute_3dTrConv_out_dim(int(output_dim[1]/2), 5)
        
        self.layers.append(LayerNorm_MA([int(hidden_channels),
                                         new_H, new_W], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Conv3d(int(hidden_channels), hidden_channels, (1,5,5), padding='same', dtype=torch.float32))
        # self.layers.append(LayerNorm_MA([hidden_channels,
        #                                  new_H, new_W], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.AdaptiveAvgPool3d((None,output_dim[0],output_dim[1]))) #padding='same'
        
        self.layers.append(nn.Conv3d(hidden_channels, hidden_channels, (1,5,5), padding='same', dtype=torch.float32))
        self.layers.append(LayerNorm_MA([hidden_channels,
                                         output_dim[0], output_dim[1]], move_dim_in = 1, move_dim_out = 2))
        self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Conv3d(hidden_channels, output_channels, (1,5,5), padding='same', dtype=torch.float32))
        # self.layers.append(LayerNorm_MA([output_channels,
        #                                  output_dim[0], output_dim[1]], move_dim_in = 1, move_dim_out = 2))
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
        for batch in range(new_X.shape[0]):
            avail_X = new_X_mask[batch,:].nonzero().squeeze()
            dropout = torch.randint(0, len(avail_X), [int(len(avail_X)*p)])
            new_X_mask[batch,avail_X[dropout]] = False
            new_X[batch,avail_X[dropout],-1] = 0
        
        return new_X, new_X_mask
    
    def __init__(self,
                 embedding_dim,
                 heads,
                 hidden_channels,
                 output_channels,
                 query_HW_dim,
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
        
        self.cb_affine_and_norm = nn.Sequential(
                                    # nn.Linear(embedding_dim, embedding_dim),
                                    # nn.LayerNorm(normalized_shape = embedding_dim),
                                    # nn.LeakyReLU(),
                                    nn.Linear(embedding_dim, hidden_channels),
                                    nn.LayerNorm(normalized_shape = hidden_channels, elementwise_affine=False),
                                    nn.LeakyReLU())
        
        self.cb_conv = nn.Sequential(nn.Conv2d(hidden_channels,
                                               hidden_channels,
                                               5,
                                               padding='same'),
                                     LayerNorm_MA((hidden_channels, *query_HW_dim)),
                                     #nn.BatchNorm2d(hidden_channels),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(hidden_channels,
                                               hidden_channels,
                                               5,
                                               padding='same'),
                                     #nn.BatchNorm2d(hidden_channels),
                                     LayerNorm_MA((hidden_channels, *query_HW_dim)),
                                     nn.LeakyReLU(),
                                     #nn.AvgPool2d(kernel_size = 5, padding=2, stride = 1),
                                     nn.Conv2d(hidden_channels,
                                               output_channels,
                                               5,
                                               padding='same'),
                                     nn.Tanh())
        
    def forward(self, X, Z, X_mask):
            
            if self.training is True and self.densification_dropout_p>0:
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
            
            target_Icond = self.cb_affine_and_norm(target_Icond)
            #target_Icond = self.cb_fc(target_Icond)
            #target_Icond = self.cb_layer_norm_2(target_Icond)
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
                 HW_dimensions,
                 kernel_size=5,
                 padding="same",
                 stride=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.HW_dimensions = HW_dimensions
        self.conv = self._make_layer(input_channles+hidden_channels, hidden_channels*4,
                                       kernel_size, padding, stride)
        
        self.value_activation = nn.Tanh() #nn.LeakyReLU()
        

    def _make_layer(self, input_channles, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(input_channles, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
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

        return torch.stack(outputs, dim = 2) # (N, C, D, H, W)
    

class ConvLSTMBlock_TF(nn.Module):
    def __init__(self, 
                 input_channles,
                 hidden_channels,
                 HW_dimensions,
                 kernel_size=5,
                 padding="same",
                 stride=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.HW_dimensions = HW_dimensions
        self.conv = nn.Conv2d(input_channles+hidden_channels, hidden_channels*4,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        
        self.linear = nn.Linear(hidden_channels, 1)
        
        self.value_activation = nn.Tanh() #nn.LeakyReLU()

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
        
        ### FROM Previous TL
        # for tstep in range(self.timestep):
            
        #     lstm_output, lstm_hidden = self.weather_lstm(
        #                                         torch.cat([weather_block_out[:,tstep,:].unsqueeze(1),
        #                                                     h_lstm_input], dim = -1),
        #                                         lstm_hidden)
        #     lstm_output = self.fc(lstm_output)            
        #     lstm_outputs.append(lstm_output)
            
        #     if y is not None: 
        #         h_lstm_input = y[0][:,tstep][:,None, None]
                
        #         h_lstm_input_mask = ~y[1][:,tstep][:,None, None]
        #         if (h_lstm_input_mask).any():
        #             h_lstm_input[h_lstm_input_mask] = lstm_output.detach()[h_lstm_input_mask]
        #     else:
        #         h_lstm_input = lstm_output.detach()
                
                
        # lstm_outputs = torch.cat(lstm_outputs, dim=1)
        
        ### FROM Soil Moisture
        # lstm_input = x[:,0,:].unsqueeze(1) # initial condition must be provided
        #     lstm_outputs = []
        #     lstm_hidden = None
            
        #     for tstep in range(0, x.shape[1]):
                
        #         lstm_output, lstm_hidden = self.lstm(lstm_input,
        #                                              lstm_hidden)
        #         lstm_output = self.linear(lstm_output)
        #         lstm_outputs.append(lstm_output)
                
                
        #         if tstep < x.shape[1]-1:
        #             if self.training is True:
        #                 nan_instances = x[:,tstep+1,self.noutputs].isnan()
        #                 lstm_input = torch.where(nan_instances[:,None,None],
        #                                           torch.concat([lstm_output.detach(),
        #                                                         x[:,tstep+1,(self.noutputs):].unsqueeze(1)], dim = -1),
        #                                           x[:,tstep+1,:].unsqueeze(1))
                        
                        
        #             else:
        #                 lstm_input = torch.concat([lstm_output.detach(), 
        #                                         x[:,tstep+1,(self.noutputs):].unsqueeze(1)],
        #                                         dim = -1)
        #         else:    
        #             break

        return torch.stack(outputs, dim = 2) # (N, C, D, H, W)
    
    
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
            setattr(self, f"activation_1_{i}",
                    nn.LeakyReLU()),
            
            
    def forward(self, input):
        outputs = []
        for i in range(self.n_channels):
            output = getattr(self, f"fc_0_{i}")(input)
            output = getattr(self, f"activation_0_{i}")(output)
            output = getattr(self, f"fc_1_{i}")(output)
            output = getattr(self, f"activation_1_{i}")(output)
            
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim = 1)
        
        return outputs
        
        
    
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
                 upsampling_dim = [104, 150]):
        
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
                 output_channels = self.convlstm_units,
                 query_HW_dim = self.upsampling_dim,
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
        
        self.convLSTM_1 = ConvLSTMBlock(input_channles = self.convlstm_input_units,
                 hidden_channels = self.convlstm_units,
                 HW_dimensions = self.upsampling_dim,
                 kernel_size=self.convlstm_kernel)
        
        self.Date_Conditioning_Module_sm = Date_Conditioning_Block(self.convlstm_units)
        self.Layer_Norm = LayerNorm_MA([self.convlstm_units,*self.upsampling_dim], 1, 2)
        
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
        Output = self.Layer_Norm(Output)
        
        Output = self.convLSTM_2(Output,
                                 target_Icond,
                                 target_Icond)
        
        Output = self.convLSTM_3(Output,
                                 target_Icond,
                                 target_Icond)
        
        Output = self.linear(torch.moveaxis(Output, 1, -1))
        #Output = self.linear_2(Output)
        
        Output = torch.moveaxis(Output, -1, 1)
        
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
    