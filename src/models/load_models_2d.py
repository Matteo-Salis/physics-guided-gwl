import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.load_2d_meteo_wtd import DiscreteDataset

############## MODEL 1 ##############
class ConvTransposeBlock(nn.Module):
    def __init__(self):
        super(ConvTransposeBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(10, 16, (1,3,3), stride=(1,3,3), dtype=torch.float32),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, (1,3,3), stride=(1,3,3), dtype=torch.float32),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.case_1d_block = nn.Sequential(
            nn.Conv3d(16, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
    
    def forward(self, x, case_1d = False):
        if case_1d:
            return self.case_1d_block(self.block(x))
        else:
            return self.block(x)
    

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(3, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),                
        )
        self.case_1d_block = nn.Sequential(
            nn.Conv3d(16, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )
    
    def forward(self, x, case_1d = False):
        if case_1d:
            return self.case_1d_block(self.block(x))
        else:
            return self.block(x)
    

# class ConvBlockFinal(nn.Module):
#     def __init__(self):
#         super(ConvBlockFinal, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv3d(32, 32, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.Conv3d(32, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#             nn.Conv3d(16, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#         )
#         self.case_1d_block = nn.Sequential(
#             nn.Conv3d(2, 2, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#             nn.BatchNorm3d(2),
#             nn.ReLU(),
#             nn.Conv3d(2, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#             nn.BatchNorm3d(1),
#             nn.ReLU(),
#             nn.Conv3d(1, 1, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
#         ) 
    
#     def forward(self, x, case_1d = False):
#         if case_1d:
#             return self.case_1d_block(x)
#         else:
#             return self.block(x)
        
class ConvBlockFinal(nn.Module):
    def __init__(self):
        super(ConvBlockFinal, self).__init__()
        self.block = nn.Sequential(
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(32, 32, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(32, 32, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(32, 16, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(16, 16, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(16, 1, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
        )
        self.case_1d_block = nn.Sequential(
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(2, 4, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(4, 8, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(8, 4, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(4, 2, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(2, 1, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
            nn.BatchNorm3d(1),
            nn.ReLU(),
            nn.ZeroPad3d((0,0,0,0,2,0)),
            nn.Conv3d(1, 1, (3,3,3), stride=(1,1,1), dtype=torch.float32, padding=(0,1,1)),
        ) 
    
    def forward(self, x, case_1d = False):
        if case_1d:
            return self.case_1d_block(x)
        else:
            return self.block(x)
    
    

class Discrete2DConcat16(nn.Module):
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
        concat = torch.concat([x_init, x_weather], dim=1)

        out = self.m_conv_f_3(concat, case_1d = True)
        
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
            nn.ReLU(),
            nn.Conv3d(16, 32, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, (1,3,3), stride=(1,3,3), dtype=torch.float32),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 16, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 10, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(10),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    

class ConvBlockSumFinal(nn.Module):
    def __init__(self):
        super(ConvBlockSumFinal, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(11, 8, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 4, (1,3,3), stride=(1,1,1), dtype=torch.float32, padding='same'),
            nn.BatchNorm3d(4),
            nn.ReLU(),
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
    