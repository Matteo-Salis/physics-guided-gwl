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

from torch.autograd import Variable

from st_moe_pytorch import MoE
from st_moe_pytorch.st_moe_pytorch import Experts
from st_moe_pytorch.st_moe_pytorch import exists
from st_moe_pytorch.st_moe_pytorch import default
from st_moe_pytorch.st_moe_pytorch import ModuleList
from st_moe_pytorch.st_moe_pytorch import AllGather
import torch.distributed as dist
from st_moe_pytorch import SparseMoEBlock

#################
### Utilities ###
#################

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
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
            
class Embedding(nn.Module):
    """
    MLP Embedding
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
    
    
# class Topographical_Embdedding(nn.Module):
    
#     def __init__(self, embedding_dim, activation):
#         super().__init__()
        
#         self.activation = activation
        
#         topo_embeddings = []
#         topo_embeddings.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
#         topo_embeddings.append(self.activation)
#         self.topo_embeddings = nn.Sequential(*topo_embeddings)
        
#     def forward(self, input):
        
#         return self.topo_embeddings(input)

########################
### Attention Blocks ###
########################

class Spatial_MHA_Block(nn.Module):
    
    def __init__(self,
                embedding_dim,
                heads,
                output_channels,
                activation,
                elementwise_affine,
                ):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        
        self.multihead_att = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        self.norm_linear = nn.Sequential(
                                    nn.LayerNorm(normalized_shape = embedding_dim,
                                                 elementwise_affine=self.elementwise_affine),
                                    nn.Linear(embedding_dim, embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, output_channels),
                                    )
        
    def forward(self, K, V, Q, attn_mask = None):
            
            output, _ = self.multihead_att(
                                        query = Q, #(N,L,E)
                                        key = K,
                                        value = V,
                                        attn_mask = attn_mask #(N,L,S) True is Not Allowed
                                        )
            output_skip = output
            output = self.norm_linear(output)
            output += output_skip
            
            self.activation(output)

            return output

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

        
        self.norm_layer_2 = nn.LayerNorm(normalized_shape = embedding_dim,
                                         elementwise_affine = self.elementwise_affine)
        
        self.mlp = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, embedding_dim),
                                    )
        
    def forward(self, input, attn_mask = None, mc_dropout = False):
        
        skip_1 = input
        output = self.norm_layer_1(input)
        
        output, _ = self.mha(
                            query = input, #(N,L,E)
                            key = input,
                            value = input,
                            attn_mask = attn_mask,
                            is_causal = True if attn_mask is not None else False
                            )
        
        output +=  skip_1
        skip_2 = output 
        
        output = self.norm_layer_2(output)
        output = self.mlp(output)
        
        output += skip_2
        
        if self.dropout_p>0:
            output = nn.functional.dropout1d(output.permute((0,2,1)),
                                            p = self.dropout_p, training = self.training or mc_dropout)
            
            output = output.permute((0,2,1))
            
        output = self.activation(output)
        
        return output

class MoE_alt(MoE):
    # inspired by https://github.com/pytorch/pytorch/issues/1333

    def __init__(self,
            dim,
            num_experts = 16,
            expert_hidden_mult = 4,
            threshold_train = 0.2,
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.,
            gating_top_n = 2,
            balance_loss_coef = 1e-2,
            router_z_loss_coef = 1e-3,
            experts: nn.Module | None = None,
            straight_through_dispatch_tensor = True,
            differentiable_topk = False,
            differentiable_topk_fused = True,
            is_distributed = None,
            allow_var_seq_len = False):      # loss weight for router z-loss):

        super(MoE_alt, self).__init__(
            dim = dim,
            num_experts = num_experts,
            gating_top_n = gating_top_n,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval,
            balance_loss_coef = balance_loss_coef,
            router_z_loss_coef = router_z_loss_coef)
        
        experts = default(experts, lambda: [Expert_alt(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])
        self.experts = Experts(
            experts,
            is_distributed = is_distributed,
            allow_var_seq_len = allow_var_seq_len
        )
        
    def forward(self, x,
        noise_gates = False,
        noise_mult = 1.):
            
        return super(MoE_alt, self).forward(x, noise_gates, noise_mult)
        

class Expert_alt(nn.Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = nn.Sequential(
            #RMSNorm(dim) if prenorm else None,
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_hidden),
            nn.LeakyReLU(),
            #GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)
        #print("Alt expert")

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        #print("Alt expert")
        return self.net(x)

class Spatial_Attention_Block_MoE(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 input_channles,
                 heads,
                 num_experts,
                 output_channels,
                 activation,
                 elementwise_affine,
                 STMoE_prenorm = False,
                 Topo_Embedder = None):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        self.num_experts = num_experts
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        if Topo_Embedder is None:
            topo_embeddings = []
            topo_embeddings.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
            topo_embeddings.append(self.activation)
            self.topo_embeddings = nn.Sequential(*topo_embeddings)
        else:
            self.topo_embeddings = Topo_Embedder
        
        value_embeddings = []
        value_embeddings.append(nn.Linear(input_channles, embedding_dim))
        value_embeddings.append(self.activation)
        self.value_embeddings = nn.Sequential(*value_embeddings)
        
        self.cb_multihead_att_1 = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        moe = MoE_alt(
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
        
            return output, total_aux_loss, balance_loss, router_z_loss
        
##############
### Models ###
##############

   
class ST_MultiPoint_Net(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                joint_mod_blocks = 1,
                joint_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                dropout = 0.2, 
                activation = "GELU"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.joint_mod_blocks = joint_mod_blocks
        self.joint_mod_heads = joint_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        # self.densification_dropout_p = densification_dropout
        self.dropout = dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        ### Joint Modules #####
        
        self.Linear = nn.Sequential(nn.Linear(int(self.embedding_dim*(sum(self.GW_W_temp_dim))),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        for i in range(self.joint_mod_blocks):
            setattr(self, f"Joint_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.joint_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = self.dropout))
            
        
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim, 1))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
        Weather_values = self.Value_Embedding_Weather(Weather_values_input)
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            attn_mask = X[2][:,GW_lag,:][:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values[:,GW_lag,:,:],
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        # Displacement modules
        Output = torch.cat([GW_out.flatten(-2,-1),
                                  Weather_out.flatten(-2,-1)], dim = -1)
        
        Output = self.Linear(Output)
        
        for i in range(self.joint_mod_blocks):
            
                Output = getattr(self, f"Joint_Module_{i}")(Output,
                                                        mc_dropout = self.training or mc_dropout)
        
        # Skip connection with last observation
        
        Output = self.Output(Output)
        
        return Output.squeeze()

class ST_MultiPoint_DisNet(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                dropout = 0.2, 
                activation = "GELU"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.displacement_mod_blocks = displacement_mod_blocks
        self.displacement_mod_heads = displacement_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        # self.densification_dropout_p = densification_dropout
        self.dropout = dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        ### Displacement Modules #####
        
        self.Linear = nn.Sequential(nn.Linear(int(self.embedding_dim*(sum(self.GW_W_temp_dim))),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = self.dropout))
            
        
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim, 1))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
        Weather_values = self.Value_Embedding_Weather(Weather_values_input)
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            attn_mask = X[2][:,GW_lag,:][:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values[:,GW_lag,:,:],
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        # Displacement modules
        Displacement = torch.cat([GW_out.flatten(-2,-1),
                                  Weather_out.flatten(-2,-1)], dim = -1)
        
        Displacement = self.Linear(Displacement)
        
        for i in range(self.displacement_mod_blocks):
            
                Displacement = getattr(self, f"Displacement_Module_{i}")(Displacement,
                                                                         mc_dropout = self.training or mc_dropout)
        
        # Skip connection with last observation
        
        Output = GW_out[:,:,:,-1] + Displacement
        
        Output = self.Output(Output)
        
        return Output.squeeze()        


#### Physics Constraint Displacement
### inspired by: https://github.com/emited/flow/tree/master

class DenseGridGen(nn.Module):

    def __init__(self, transpose=True):
        super(DenseGridGen, self).__init__()
        self.transpose = transpose
        self.register_buffer('grid', torch.Tensor())

    def forward(self, x):

        if self.transpose:
            x = x.transpose(1, 2).transpose(2, 3)

        g0 = torch.linspace(-1, 1, x.size(2)
                            ).unsqueeze(0).repeat(x.size(1), 1)
        g1 = torch.linspace(-1, 1, x.size(1)
                            ).unsqueeze(1).repeat(1, x.size(2))
        grid = torch.cat([g0.unsqueeze(-1), g1.unsqueeze(-1)], -1)
        self.grid.resize_(grid.size()).copy_(grid)

        bgrid = Variable(self.grid)
        bgrid = bgrid.unsqueeze(0).expand(x.size(0), *bgrid.size())

        return bgrid - x

class GaussianWarpingScheme(nn.Module):
    def __init__(self, padding_mode='zeros', F=3, std=0.25):
        super(GaussianWarpingScheme, self).__init__()
        self.grid = DenseGridGen()
        self.F = F
        self.std = std
        self.padding_mode = padding_mode

    def forward(self, im, w):
        return F.grid_sample(im, self.grid(w), padding_mode=self.padding_mode, mode='gaussian', F=self.F, std=self.std)

class ST_MultiPoint_PhyDisNet(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                dropout = 0.2, 
                activation = "GELU"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.displacement_mod_blocks = displacement_mod_blocks
        self.displacement_mod_heads = displacement_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.dropout = dropout
        self.activation = activation
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = False)
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        ### Displacement Modules #####
        
        self.Linear = nn.Sequential(nn.Linear(int(self.embedding_dim*(sum(self.GW_W_temp_dim))),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = self.dropout))
            
            
        self.Warping_Scheme = GaussianWarpingScheme()
            
        
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim, 1))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
        Weather_values = self.Value_Embedding_Weather(Weather_values_input)
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            attn_mask = X[2][:,GW_lag,:][:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values[:,GW_lag,:,:],
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        # Displacement modules
        Displacement = torch.cat([GW_out.flatten(-2,-1),
                                  Weather_out.flatten(-2,-1)], dim = -1)
        
        Displacement = self.Linear(Displacement)
        
        for i in range(self.displacement_mod_blocks):
            
                Displacement = getattr(self, f"Displacement_Module_{i}")(Displacement,
                                                                         mc_dropout = self.training or mc_dropout)
        
        
        # Skip connection with last observation
        
        Output = GW_out[:,:,:,-1] + Displacement
        
        Output = self.Warping_Scheme(GW_out[:,:,:,-1].unsqueeze(1),
                                     Displacement)
        
        Output = self.Output(Output)
        
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
    