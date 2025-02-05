import json
import math

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.load_2d_meteo_wtd import DiscreteDataset

class ConvTransposeBlock(nn.Module):
    def __init__(self, filters_in = 10, filters_out = 16, conv_num = 2):
        super(ConvTransposeBlock, self).__init__()
        
        self.layers = []
        self.layers.append(nn.ConvTranspose3d(filters_in, 16, (1,3,3), stride=(1,3,3), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(16))
        self.layers.append(nn.LeakyReLU())
        # expanding filters
        for i in range(conv_num):
            self.layers.append(nn.Conv3d(16*(2**(i)), 16*(2**(i+1)), (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
            self.layers.append(nn.BatchNorm3d(16*(2**(i+1))))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.ConvTranspose3d(16*(2**conv_num), 16*(2**conv_num), (1,3,3), stride=(1,3,3), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(16*(2**conv_num)))
        self.layers.append(nn.LeakyReLU())
        # reducing filters
        for i in range(conv_num):
            self.layers.append(nn.Conv3d(16*(2**(conv_num-i)), 16*(2**(conv_num-i-1)), (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
            self.layers.append(nn.BatchNorm3d(16*(2**(conv_num-i-1))))
            self.layers.append(nn.LeakyReLU())

        self.layers.append(nn.Conv3d(16, filters_out, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
        self.layers.append(nn.BatchNorm3d(filters_out))
        self.layers.append(nn.LeakyReLU())
        
        self.block = nn.Sequential(*self.layers)
        
    
    def forward(self, x):
        return self.block(x)
    

class ConvBlock(nn.Module):
    def __init__(self, filters_in = 3, filters_out = 16, conv_num = 5):
        super(ConvBlock, self).__init__()

        self.layers = []
        self.layers.append(nn.Conv3d(filters_in, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
        self.layers.append(nn.BatchNorm3d(8))
        self.layers.append(nn.LeakyReLU())
        # expanding filters
        for i in range(conv_num):
            self.layers.append(nn.Conv3d(8*(2**(i)), 8*(2**(i+1)), (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
            self.layers.append(nn.BatchNorm3d(8*(2**(i+1))))
            self.layers.append(nn.LeakyReLU())
        # reducing filters
        for i in range(conv_num):
            self.layers.append(nn.Conv3d(8*(2**(conv_num-i)), 8*(2**(conv_num-i-1)), (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
            self.layers.append(nn.BatchNorm3d(8*(2**(conv_num-i-1))))
            self.layers.append(nn.LeakyReLU())
            if 8*(2**(conv_num-i-1)) == filters_out:
                self.block = nn.Sequential(*self.layers)
                return None

        self.layers.append(nn.Conv3d(8, filters_out, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'))
        self.layers.append(nn.BatchNorm3d(filters_out))
        self.layers.append(nn.LeakyReLU())
        
        self.block = nn.Sequential(*self.layers)
  
    def forward(self, x):
        return self.block(x)
    
        
class ConvBlockFinal(nn.Module):
    def __init__(self, input_ch = 19, k = 3, conv_num = 5, del_time_block = False):
        super(ConvBlockFinal, self).__init__()

        self.layers = []

        pow_2 = int(math.log2(input_ch))
        filters_pow = 2**(pow_2+1)
        if filters_pow < 4:
            filters_pow = 4

        self.layers.append(nn.ReplicationPad3d((1,1,1,1,k-1,0)))
        self.layers.append(nn.Conv3d(input_ch, filters_pow, (k,3,3), stride=(1,1,1), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(filters_pow))
        self.layers.append(nn.LeakyReLU())

        # expanding filters
        for i in range(conv_num):
            if filters_pow <= 4*(2**(i)):
                self.layers.append(nn.ReplicationPad3d((1,1,1,1,k-1,0)))
                self.layers.append(nn.Conv3d(4*(2**(i)), 4*(2**(i+1)), (k,3,3), stride=(1,1,1), dtype=torch.float32))
                self.layers.append(nn.BatchNorm3d(4*(2**(i+1))))
                self.layers.append(nn.LeakyReLU())

        for i in range(conv_num):
            self.layers.append(nn.ReplicationPad3d((1,1,1,1,k-1,0)))
            self.layers.append(nn.Conv3d(4*(2**(conv_num-i)), 4*(2**(conv_num-i-1)), (k,3,3), stride=(1,1,1), dtype=torch.float32))
            self.layers.append(nn.BatchNorm3d(4*(2**(conv_num-i-1))))
            self.layers.append(nn.LeakyReLU())
        
        if del_time_block:
            self.layers.append(nn.ReplicationPad3d((1,1,1,1,k-2,0)))
        else:
            self.layers.append(nn.ReplicationPad3d((1,1,1,1,k-1,0)))
        self.layers.append(nn.Conv3d(4, 1, (k,3,3), stride=(1,1,1), dtype=torch.float32))
        self.layers.append(nn.BatchNorm3d(1))
        self.layers.append(nn.LeakyReLU())

        self.block = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.block(x)
    
    
############## MODEL 1 ##############
class Discrete2DConcat16(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        self.m_conv_1 = ConvBlock(conv_num=3, filters_out = 16)

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock(filters_out = 16)
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockFinal(input_ch=32, k=3)


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = self.m_conv_1(x_init)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat)
        
        return out
    
############## MODEL 2 ##############
class Discrete2DConcat1(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        self.m_conv_1 = ConvBlock(conv_num=3, filters_out = 1)

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock(filters_out = 1)
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockFinal(input_ch=2, k=3, conv_num=5)


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = self.m_conv_1(x_init)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat)
        
        return out
    
############## MODEL 2 BIS ##############
class Discrete2DConcat1_Time(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        self.m_conv_1 = ConvBlock(conv_num=2, filters_out = 1)

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock(conv_num = 2, filters_out = 1)
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockFinal(input_ch=1, k=3, conv_num=2, del_time_block = True)


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = self.m_conv_1(x_init)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        concat = torch.concat([x_init, x_weather], dim=2)

        out = self.m_conv_f_3(concat)
        
        return out

############## MODEL 3 ##############

class Discrete2DConcat1Sum(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        self.m_conv_1 = ConvBlock()

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock()
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockFinal()


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = self.m_conv_1(x_init, case_1d = True)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather, case_1d = True)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        sum = torch.add(x_init, x_weather)

        # test
        out = sum

        # TODO: combine the sum with convolutions
        # out = self.m_conv_f_3(sum, case_1d = True)
        
        return out

############## MODEL 4 ##############

class ConvBlockSuperRes(nn.Module):
    def __init__(self):
        super(ConvBlockSuperRes, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(10, 16, (1,3,3), stride=(1,3,3), dtype=torch.float32),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose3d(64, 32, (1,3,3), stride=(1,3,3), dtype=torch.float32),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 10, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(10),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    

class ConvBlockSumFinal(nn.Module):
    def __init__(self):
        super(ConvBlockSumFinal, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(11, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 4, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(),
            nn.Conv3d(4, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
        )
    
    def forward(self, x):
        return self.block(x)


class Discrete2DSumSuperRes(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        self.m_conv_1 = ConvBlock()

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvBlockSuperRes()
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockSumFinal()


    def forward(self, x, training = True):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = self.m_conv_1(x_init, case_1d = True)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)

        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat)

        # out[:,0,:,:,:] = out[:,0,:,:,:] + x_weather[:,4,:,:,:] # adding prain
        # out[:,0,:,:,:] = out[:,0,:,:,:] + x_weather[:,9,:,:,:] # adding snowmelt
        # out[:,0,:,:,:] = out[:,0,:,:,:] - x_weather[:,6,:,:,:] # substracting et
  
        if training:
            return out, x_weather
        
        return out
    

############## MODEL 5 ##############

class Discrete2DVanillaConcat(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock()
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvBlockFinal(k = 2)


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat, case_preproc = True)
        
        return out

###########################################

# Implementation from https://github.com/czifan/ConvLSTM.pytorch/blob/master/networks/ConvLSTM.py (to cite)
class ConvLSTMBlock(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.num_features = num_features
        self.conv = self._make_layer(in_channels+num_features, num_features*4,
                                       kernel_size, padding, stride)

    def _make_layer(self, in_channels, out_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, inputs):
        '''

        :param inputs: (B, S, C, H, W)
        :param hidden_state: (hx: (B, S, C, H, W), cx: (B, S, C, H, W))
        :return:
        '''
        outputs = []
        B, C, S, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        for t in range(S):
            combined = torch.cat([inputs[:,:,t,:,:], # (B, C, H, W)
                                  hx], dim=1)
            gates = self.conv(combined)
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 2, 0, 3, 4).contiguous() # (S, B, C, H, W) -> (B, S, C, H, W)

############## MODEL 6 ##############

class Discrete2DConvLSTM(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # transpose conv (upsampling weather)
        self.m_conv_tr_1 = ConvTransposeBlock()
        self.m_avg_pool_2b = nn.AdaptiveMaxPool3d((None, 57, 84))

        self.m_conv_f_3 = ConvLSTMBlock(19, 8)
        self.m_conv_f_4 = ConvLSTMBlock(8, 1)



    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, dtm, x_weather) = x 

        # processing initial values x
        dtm = torch.unsqueeze(dtm, dim=0)
        dtm = dtm.expand(x_init.shape[0],-1,-1,-1)
        x_init = torch.concat([x_init, dtm], dim=1)
        x_init = torch.unsqueeze(x_init, dim=2)

        # processing weaterh
        x_weather = self.m_conv_tr_1(x_weather)
        x_weather = self.m_avg_pool_2b(x_weather)

        # concat
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat)
        out = self.m_conv_f_4(out)
        
        return out
###########################################


    
if __name__ == "__main__":
    print("Loading data.json...")
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/discrete_2D_wtd/test_2D_blocks.json') as f:
        dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

    print("Loading DiscreteDataset...")
    ds = DiscreteDataset(dict_files)
    wtd_init, weather, wtd_out  = ds[0]
    x = [wtd_init, weather]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Loading Discrete2DNN...")
    timesteps = dict_files["timesteps"]
    model = Discrete2DNN(timesteps).to(device)
    print("Discrete2DNN prediction...")
    y = model(x)
    print(f"Output:\n{y}")
    