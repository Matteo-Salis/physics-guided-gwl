import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.load_2d_meteo_wtd import DiscreteDataset

class Discrete2DNN(nn.Module):
    def __init__(self, timesteps = 180, num_layers = 1):
        super().__init__()
        
        self.timesteps = timesteps

        # processing initial values
        # self.conv1_init = nn.Conv2d(1, 1, 3) # ch_in, ch_out, kernel

        # transpose conv (upsampling weather)
        self.bn_1 = nn.BatchNorm3d(10)
        self.conv1_w = nn.ConvTranspose3d(10, 10, (1,2,2), stride=(1,2,2), dtype=torch.float32)
        self.bn_2 = nn.BatchNorm3d(10)
        self.conv2_w = nn.ConvTranspose3d(10, 10, (1,2,2), stride=(1,2,2), dtype=torch.float32)
        self.bn_3 = nn.BatchNorm3d(10)
        self.conv3_w = nn.ConvTranspose3d(10, 10, (1,2,2), stride=(1,2,2), dtype=torch.float32)
        self.bn_4 = nn.BatchNorm3d(10)
        self.conv4_w = nn.ConvTranspose3d(10, 10, (1,2,2), stride=(1,2,2), dtype=torch.float32)
        self.avg_pool_5 = nn.AdaptiveAvgPool3d((None, 114, 167))

        self.conv_6 = nn.Conv3d(12, 1, 1, stride=1, dtype=torch.float32)


    def forward(self, x):
        """
        return 
            lstm_out (array): lstm_out = [S_we, M, P_r, Es, K_s, K_r]
        x: tensor of shape (L,Hin) if minibatches itaration (L,N,Hin) when batch_first=False (default)
        """
        (x_init, x_weather) = x 

        # processing initial values x
        # x_init = F.relu(self.conv1_init(x_init))

        # processing weaterh
        x_weather = F.relu(self.bn_1(self.conv1_w(x_weather)))
        x_weather = F.relu(self.bn_2(self.conv2_w(x_weather)))
        x_weather = F.relu(self.bn_3(self.conv3_w(x_weather)))
        x_weather = F.relu(self.bn_4(self.conv4_w(x_weather)))
        x_weather = F.relu(self.avg_pool_5(x_weather))

        # concat
        # print(x_init.shape)
        x_init = torch.unsqueeze(x_init, dim=2)
        x_init = x_init.expand(-1,-1,self.timesteps,-1,-1)
        x = torch.concat([x_init, x_weather], dim=1)
        out = F.relu(self.conv_6(x))
        
        return out
    
if __name__ == "__main__":
    print("Loading data.json...")
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/data.json') as f:
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
    