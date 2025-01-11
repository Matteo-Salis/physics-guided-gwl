import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.load_1d_meteo_wtd import ContinuousDataset

class Continuous1DNN_idw(nn.Module):
    
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
        cb_fc.append(nn.ReLU())
        for l in range(self.cb_fc_layer - 2):
            cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
            cb_fc.append(nn.ReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(14, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.ReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.lstm_input_units, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.lstm_input_units))
        conv3d_stack.append(nn.ReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(8, 1))
        self.fc = nn.Sequential(*fc)
        
    def idw(self, dist, values, x_mask, weight_std = True):
        
        weights = 1/dist
        weights = torch.minimum(weights, torch.tensor([1e8]).expand(weights.shape).to(weights.device))
        weights = weights * x_mask 
        numerator = torch.sum(weights*values, dim = 1)
        denominator = torch.sum(weights, dim = 1)
        output = numerator/denominator
        weights_cv = torch.std(weights) / torch.mean(weights)
            
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
                             target0[1].expand((z.shape[0], 1))], dim = -1)
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_sim = w[1] * z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_sim = torch.sum(weather_sim, dim = -1)/torch.sqrt(torch.tensor(weather_sim.shape[-1]))
        
        weather_sim = weather_sim[:, None, None, : ,:].expand(-1, -1, w[0].shape[2], -1, -1 )
        
        
        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             weather_sim], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        wb_td3dconv = wb_td3dconv.squeeze((3,4))
        wb_td3dconv = torch.moveaxis(wb_td3dconv, 1, -1)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = nn.Tanh()(torch.movedim(target0_h, 0, 1))
        
        target_ts = self.lstm_1(wb_td3dconv,
                                 (target0_h.contiguous(),
                                  torch.zeros_like(target0_h))) #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        target0_skip = target0.unsqueeze(1).expand([-1,self.timestep,-1])
        target_ts_out = target_ts[0] + target0_skip
        
        target_ts_out = self.fc(target_ts_out)
        
        return target_ts_out.squeeze()
    

############## MODEL 2 ##############

class Continuous1DNN_dweight(nn.Module):
    def __init__(self,
                 timestep = 180,
                 cb_fc_layer = 5,
                 cb_fc_neurons = 32,
                 cb_fc_w0_layer = 3,
                 cb_fc_w0_neurons = 16,
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
        self.cb_fc_w0_layer = cb_fc_w0_layer
        self.cb_fc_w0_neurons = cb_fc_w0_neurons
        self.conv_filters = conv_filters
        
        cb_fc_w0 = []
        cb_fc_w0.appen(nn.Linear(4, self.cb_fc_w0_neurons))
        for l in range(self.cb_fc_layer - 2):
            cb_fc_w0.append(nn.Linear(self.cb_fc_w0_neurons, self.cb_fc_w0_neurons))
            cb_fc_w0.append(nn.ReLU())
        cb_fc_w0.append(nn.Linear(self.cb_fc_w0_neurons, 1))
        self.cb_fc_w0 = nn.Sequential(*cb_fc_w0)
        
        
        self.w0gamma = nn.Softmax(dim = 1)
        # Fully connected
        cb_fc = []
        cb_fc.append(nn.Linear(5, self.cb_fc_neurons))
        cb_fc.append(nn.ReLU())
        for l in range(self.cb_fc_layer - 2):
            cb_fc.append(nn.Linear(self.cb_fc_neurons, self.cb_fc_neurons))
            cb_fc.append(nn.ReLU())
        
        cb_fc.append(nn.Linear(self.cb_fc_neurons, self.lstm_units))
        #cb_fc.append(nn.ReLU())
        self.cb_fc = nn.Sequential(*cb_fc)
        
        # Weather block
        #self.weather_wgamma = nn.Softmax(dim = -1)
        
        conv3d_stack=[]
        conv3d_stack.append(nn.Conv3d(14, self.conv_filters, (1,2,2))) # Conv input (N, C, D, H, W) - kernel 3d (D, H, W)
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        
        for i in range(4):
            conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
            conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
            conv3d_stack.append(nn.ReLU())
            
        conv3d_stack.append(nn.AdaptiveAvgPool3d((None,4,4)))
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.conv_filters, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.conv_filters))
        conv3d_stack.append(nn.ReLU())
        conv3d_stack.append(nn.Conv3d(self.conv_filters, self.lstm_input_units, (1,2,2)))
        conv3d_stack.append(nn.BatchNorm3d(self.lstm_input_units))
        conv3d_stack.append(nn.ReLU())
        self.conv3d_stack = nn.Sequential(*conv3d_stack)
            
        # Joint sequental block
        self.lstm_1 = nn.LSTM(self.lstm_input_units, self.lstm_units,
                              batch_first=True,
                              num_layers=self.lstm_layer) # Batch first input (N,L,H)
        
        fc = []
        fc.append(nn.Linear(self.lstm_units, 8))
        fc.append(nn.ReLU())
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
        
        # target_sim = torch.matmul(x[:,:,:3], z.unsqueeze(-1))
        # target_sim = target_sim/torch.sqrt(torch.tensor(z.shape[-1]))
        target_dist = torch.cdist(x[:,:,:3], z.unsqueeze(1), p=2.0) # (B×P×M), (B×R×M), OUTPUT: (B×P×R)
        target0_weight = torch.cat([x[:,:,:3], target_dist.unsqueeze(1)], dim = -1)
        print(target0_weight[:,:5,:]) 
        
        target0_weight = self.cb_fc_w0(target0_weight)
        target0_weight = target0_weight * x_mask # masking nan
        print(target0_weight[:,:5]) 
        target0_weight = self.w0gamma(target0_weight)
        print(target0_weight[:,:5]) 
        
        
        #target_sim = self.wgamma(target_sim.squeeze(-1))
        #print(target_sim[:,:10])
        #print("sum", torch.sum(target_sim, dim = 1))
        
        #print(target_sim[:,:10])
        #print("sum", torch.sum(target_sim, dim = 1))
        target_sim_sum = torch.sum(target0_weight, dim = 1)
        print("sum", target_sim_sum)
        target0 = torch.sum(x[:,:,-1] * target_sim_sum, dim = 1)
        target0 = torch.cat([z, target0.unsqueeze(-1), target_sim_sum.unsqueeze(-1)], dim = -1)
        print(target0)
        
        target0 = self.cb_fc(target0)
        
        # Weather block
        ## w[0] (10, 180, 9, 12); w[1] (9, 12, 3)
        weather_sim = w[1] * z[:,None,None,:].expand(-1, w[1].shape[1], w[1].shape[2], -1)
        weather_sim = torch.sum(weather_sim, dim = -1)/torch.sqrt(torch.tensor(weather_sim.shape[-1]))
        #weather_sim = self.weather_wgamma(weather_sim)
        
        weather_sim = weather_sim[:, None, None, : ,:].expand(-1, -1, w[0].shape[2], -1, -1 )
        
        
        weather = torch.cat([w[0],
                             torch.moveaxis(w[1], -1, 1).unsqueeze(2).expand(-1, -1, w[0].shape[2], -1, -1 ) ,
                             weather_sim], dim = 1)
        
        wb_td3dconv = self.conv3d_stack(weather)
        
        wb_td3dconv = wb_td3dconv.squeeze((3,4))
        wb_td3dconv = torch.moveaxis(wb_td3dconv, 1, -1)
        
        print(wb_td3dconv.shape)
        
        # Sequential block
        target0_h = target0.unsqueeze(1).expand([-1,self.lstm_layer,-1])
        target0_h = nn.Tanh()(torch.movedim(target0_h, 0, 1))
        
        print(target0_h.shape) 
        
        target_ts = self.lstm_1(wb_td3dconv,
                                 (target0_h.contiguous(),
                                  torch.zeros_like(target0_h))) #input  [input, (h_0, c_0)] - h and c (D∗num_layers,N,H)
        
        target0_skip = target0.unsqueeze(1).expand([-1,self.timestep,-1])
        target_ts_out = target_ts[0] + target0_skip
        
        target_ts_out = self.fc(target_ts_out)
        
        return target_ts_out.squeeze()
    


###########################################

    
if __name__ == "__main__":
    print("Loading data.json...")
    dict_files = {}
    with open('/leonardo_scratch/fast/IscrC_DL4EO/github/water-pinns/src/configs/continuous_1D_wtd/test_1D.json') as f:
        dict_files = json.load(f)
    print(f"Read data.json: {dict_files}")

    print("Loading ContinuousDataset...")
    ds = ContinuousDataset(dict_files)
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
    timesteps = dict_files["timesteps"]
    model = Continuous1DNN(timestep = dict_files["timesteps"],
                 cb_fc_layer = dict_files["cb_fc_layer"], #5,
                 cb_fc_neurons = dict_files["cb_fc_neurons"], # 32,
                 conv_filters = dict_files["conv_filters"], #32,
                 lstm_layer = dict_files["lstm_layer"], #5,
                 lstm_input_units = dict_files["lstm_input_units"], #16,
                 lstm_units = dict_files["lstm_units"] #32
                 ).to(device)
    print("Continuous1DNN prediction...")
    y = model(x, z, w, x_mask)
    print(f"Output:\n{y}")
    