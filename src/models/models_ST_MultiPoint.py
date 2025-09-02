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

# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
            
def weight_init(m, activation):
    if activation == "ReLU":
        nonlinearity='relu'
    else:     
        nonlinearity='leaky_relu'
        
    
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity = nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class Densification_Dropout(nn.Module):
    def __init__(self,
                 drop_value = 0,
                 p = 0.25):
        super().__init__()
        """
        Dropout training as in densification of Andrychowicz et al. (2023)
        sample_values: (N,S)
        sample_mask: (N,S)
        
        choose p carefully (avoid probability of all set no nan)
        """
        
        self.drop_value = drop_value
        self.p = p
    
    def forward(self, sample_values, sample_mask):
        
        new_values = sample_values.clone()
        new_mask = sample_mask.clone()
        
        for batch in range(new_values.shape[0]):
            not_nan_idxs = torch.logical_not(new_mask[batch,:]).nonzero().squeeze()
            
            dropout = torch.randint(0, len(not_nan_idxs), [int(len(not_nan_idxs)*self.p)])
            new_mask[batch,not_nan_idxs[dropout]] = True
            new_values[batch,not_nan_idxs[dropout],:] = self.drop_value
        
        return new_values, new_mask
            
class Embedding(nn.Module):
    """
    MLP Embedding
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 normalization,
                 norm_dims = "2d",
                 elementwise_affine = False):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
            
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.elementwise_affine = elementwise_affine
        
        layers = []
        
        layers.append(nn.Linear(self.in_channels, self.hidden_channels))
        layers.append(self.activation)
        
        if self.normalization is not None:
            if self.normalization == "layernorm":
                layers.append(nn.LayerNorm(self.hidden_channels,
                                           elementwise_affine = self.elementwise_affine))
            elif self.normalization == "instancenorm":
                if norm_dims == "2d":
                    layers.append(InstanceNorm2d_MA(self.hidden_channels,-1,1,
                                    elementwise_affine=self.elementwise_affine))
                elif norm_dims == "1d":
                    layers.append(InstanceNorm1d_MA(self.hidden_channels,-1,1,
                                    elementwise_affine=self.elementwise_affine))
                
        layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        layers.append(self.activation)
        
        if self.normalization is not None:
            if self.normalization == "layernorm":
                layers.append(nn.LayerNorm(self.hidden_channels,
                                           elementwise_affine = self.elementwise_affine))
            elif self.normalization == "instancenorm":
                if norm_dims == "2d":
                    layers.append(InstanceNorm2d_MA(self.hidden_channels,-1,1,
                                    elementwise_affine=self.elementwise_affine))
                elif norm_dims == "1d":
                    layers.append(InstanceNorm1d_MA(self.hidden_channels,-1,1,
                                    elementwise_affine=self.elementwise_affine))
                
        layers.append(nn.Linear(self.hidden_channels, self.out_channels))
        layers.append(self.activation)
        
        self.ST_layers = nn.Sequential(*layers)
            
    def forward(self, input):
        """
        input (B, D, S, C_in)
        output (B, D, S, C_out)
        """
        
        return self.ST_layers(input)
    

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
    
class InstanceNorm1d_MA(nn.Module):
    
    def __init__(self, normalized_shape, move_dim_from = None, move_dim_to = None, 
                 eps = 0.00001, elementwise_affine = True, device=None, dtype=None):
        super().__init__()
        
        self.norm = []
        self.norm.append(nn.InstanceNorm1d(normalized_shape, affine=elementwise_affine, eps = eps,
                                      device = device, dtype = dtype))
        
        if move_dim_from is not None:
            self.norm.insert(0, MoveAxis(move_dim_from, move_dim_to)) #partial(torch.moveaxis, source = norm_dim, destination = -1)
            self.norm.append(MoveAxis(move_dim_to, move_dim_from)) #partial(torch.moveaxis, source = -1, destination = norm_dim)
            
        self.norm = nn.Sequential(*self.norm)
        
    def forward(self, input):
            
        norm_output = self.norm(input)
        return norm_output


class InstanceNorm2d_MA(nn.Module):
    
    def __init__(self, normalized_shape, move_dim_from = None, move_dim_to = None, 
                 eps = 0.00001, elementwise_affine = True, device=None, dtype=None):
        super().__init__()
        
        self.norm = []
        self.norm.append(nn.InstanceNorm2d(normalized_shape, affine=elementwise_affine, eps = eps,
                                      device = device, dtype = dtype))
        
        if move_dim_from is not None:
            self.norm.insert(0, MoveAxis(move_dim_from, move_dim_to)) #partial(torch.moveaxis, source = norm_dim, destination = -1)
            self.norm.append(MoveAxis(move_dim_to, move_dim_from)) #partial(torch.moveaxis, source = -1, destination = norm_dim)
            
        self.norm = nn.Sequential(*self.norm)
        
    def forward(self, input):
            
        norm_output = self.norm(input)
        return norm_output    
    
class Conv3d_Embedding(nn.Module):
    """
    MLP Embedding
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 activation,
                 padding_mode = "replicate",
                 normalization = None,
                 elementwise_affine = True,
                 avg_pooling = False):
        super().__init__()
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
            
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.elementwise_affine = elementwise_affine
        self.avg_pooling = avg_pooling
        
        layers = []
        
        layers.append(nn.Conv3d(self.in_channels, self.hidden_channels, (1,5,5),
                                padding='same', padding_mode = padding_mode,
                                dtype=torch.float32))
        layers.append(self.activation)
        
        if self.normalization is not None:
            if self.normalization == "layernorm":
                layers.append(LayerNorm_MA(self.hidden_channels, move_dim_from = 1, move_dim_to = -1,
                                           elementwise_affine = self.elementwise_affine))
            elif self.normalization == "instancenorm":
                layers.append(nn.InstanceNorm3d(self.hidden_channels, affine = self.elementwise_affine))
                
        layers.append(nn.Conv3d(self.hidden_channels, self.hidden_channels, (1,5,5),
                                padding='same', padding_mode = padding_mode,
                                dtype=torch.float32))
        layers.append(self.activation)
        
        if self.normalization is not None:
            if self.normalization == "layernorm":
                layers.append(LayerNorm_MA(self.hidden_channels, move_dim_from = 1, move_dim_to = -1,
                                           elementwise_affine = self.elementwise_affine))
            elif self.normalization == "instancenorm":
                layers.append(nn.InstanceNorm3d(self.hidden_channels, affine = self.elementwise_affine))
        
        if self.avg_pooling is True:
            layers.append(nn.AvgPool3d((1,3,3),count_include_pad=False))
                            
        layers.append(nn.Conv3d(self.hidden_channels, self.hidden_channels, (1,5,5),
                                padding='same', padding_mode = padding_mode,
                                dtype=torch.float32))
        layers.append(self.activation)
        
        self.ST_layers = nn.Sequential(*layers)
            
    def forward(self, input):
        """
        input (B, D, S, C_in)
        output (B, D, S, C_out)
        """
        
        return self.ST_layers(input)

########################
### Attention Blocks ###
########################

class Spatial_MHA_Block(nn.Module):
    
    def __init__(self,
                embedding_dim,
                heads,
                output_channels,
                activation,
                normalization,
                elementwise_affine,
                ):
        super().__init__()
        
        self.normalization = normalization
        self.elementwise_affine = elementwise_affine
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()


        
        self.multihead_att = nn.MultiheadAttention(embedding_dim, heads,
                                                   batch_first=True)
        
        norm_linear_list = []
        if self.normalization is not None:
            if self.normalization == "layernorm":
                norm_linear_list.append(nn.LayerNorm(normalized_shape = embedding_dim,
                                        elementwise_affine=self.elementwise_affine))
            elif self.normalization == "instancenorm":
                norm_linear_list.append(InstanceNorm1d_MA(embedding_dim,1,-1,
                                   elementwise_affine=self.elementwise_affine))
       
        norm_linear_list.extend([self.activation,
                            nn.Linear(embedding_dim, embedding_dim),
                            self.activation,
                            nn.Linear(embedding_dim, output_channels)])
        
        self.norm_linear = nn.Sequential(*norm_linear_list)
        
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
                 normalization,
                 elementwise_affine):
        super().__init__()
        
        self.normalization = normalization
        self.elementwise_affine = elementwise_affine
        self.dropout_p = dropout_p
        
        if activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
            
        if self.normalization is not None:
            if self.normalization == "layernorm":
                self.norm_layer_1 = nn.LayerNorm(normalized_shape = embedding_dim,
                                        elementwise_affine=self.elementwise_affine)
            elif self.normalization == "instancenorm":
                self.norm_layer_1 = InstanceNorm1d_MA(embedding_dim,1,-1,
                                   elementwise_affine=self.elementwise_affine)
        
        self.mha = nn.MultiheadAttention(embedding_dim, heads,
                                         batch_first=True)

        
        if self.normalization is not None:
            if self.normalization == "layernorm":
                self.norm_layer_2 = nn.LayerNorm(normalized_shape = embedding_dim,
                                        elementwise_affine=self.elementwise_affine)
            elif self.normalization == "instancenorm":
                self.norm_layer_2 = InstanceNorm1d_MA(embedding_dim,1,-1,
                                   elementwise_affine=self.elementwise_affine)
        
        self.mlp = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, embedding_dim),
                                    )
        
    def forward(self, input, attn_mask = None, mc_dropout = False):
        
        skip_1 = input
        if self.normalization is not None:
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
        if self.normalization is not None:      
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
                emb_W = "3DCNN",
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                joint_mod_blocks = 1,
                joint_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
                dropout = 0.2, 
                activation = "GELU",
                normalization = "layernorm"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.joint_mod_blocks = joint_mod_blocks
        self.joint_mod_heads = joint_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
        self.dropout = dropout
        self.activation = activation
        self.emb_W = emb_W
        self.normalization = normalization
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        if self.emb_W == "3DCNN":
            self.Value_Embedding_Weather = Conv3d_Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        elif self.emb_W == "linear":
            self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=self.densification_dropout_dv,
                                                           p = self.densification_dropout_p)
        
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        ### Joint Modules #####
        
        linear_list = [nn.Linear(int(self.embedding_dim*(sum(self.GW_W_temp_dim))),
                                                self.embedding_dim)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                linear_list.append(nn.LayerNorm(self.embedding_dim))
            elif self.normalization == "instancenorm":
                linear_list.append(InstanceNorm1d_MA(self.embedding_dim,1,-1,
                                   elementwise_affine=True))
                
        linear_list.append(self.activation_fn)
        self.Linear = nn.Sequential(*linear_list)
        
        for i in range(self.joint_mod_blocks):
            setattr(self, f"Joint_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.joint_mod_heads,
                                self.activation,
                                normalization = self.normalization,
                                elementwise_affine = True,
                                dropout_p = 0))
            
        
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim//2,
                                              bias = False),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim//2, 1, bias = False))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        if self.emb_W == "3DCNN":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input).permute((0,2,3,4,1)).flatten(2,3)
        elif self.emb_W == "linear":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input)
            
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    X[2][:,GW_lag,:])
            
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = X[2][:,GW_lag,:]
            
            attn_mask = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        # Joint modules
        Output = torch.cat([GW_out.flatten(-2,-1),
                                  Weather_out.flatten(-2,-1)], dim = -1)
        
        Output = self.Linear(Output)
        
        for i in range(self.joint_mod_blocks):
            
                Output = getattr(self, f"Joint_Module_{i}")(Output,
                                                        mc_dropout = self.training or mc_dropout)
        
        # Skip connection with last observation
        if self.dropout > 0:
            Output = nn.functional.dropout1d(Output.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
                
            Output = Output.permute((0,2,1))
            
        Output = self.Output(Output)
        
        return Output.squeeze()
    

class ST_MultiPoint_Net_SAGW(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                emb_W = "3DCNN",
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                joint_mod_blocks = 1,
                joint_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
                dropout = 0.2, 
                activation = "GELU",
                normalization = "layernorm"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.joint_mod_blocks = joint_mod_blocks
        self.joint_mod_heads = joint_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
        self.dropout = dropout
        self.activation = activation
        self.emb_W = emb_W
        self.normalization = normalization
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        if self.emb_W == "3DCNN":
            self.Value_Embedding_Weather = Conv3d_Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        elif self.emb_W == "linear":
            self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=self.densification_dropout_dv,
                                                           p = self.densification_dropout_p)
        
        
        ### Spatial Modules #####
        self.SAGW_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        ### Joint Modules #####
        
        linear_list = [nn.Linear(int(self.embedding_dim*(sum(self.GW_W_temp_dim))),
                                                self.embedding_dim)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                linear_list.append(nn.LayerNorm(self.embedding_dim))
            elif self.normalization == "instancenorm":
                linear_list.append(InstanceNorm1d_MA(self.embedding_dim,1,-1,
                                   elementwise_affine=True))
                
        linear_list.append(self.activation_fn)
        self.Linear = nn.Sequential(*linear_list)
        
        for i in range(self.joint_mod_blocks):
            setattr(self, f"Joint_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.joint_mod_heads,
                                self.activation,
                                normalization = self.normalization,
                                elementwise_affine = True,
                                dropout_p = 0))
            
        
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim//2,
                                              bias = True),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim//2, 1, bias = True))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        if self.emb_W == "3DCNN":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input).permute((0,2,3,4,1)).flatten(2,3)
        elif self.emb_W == "linear":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input)
            
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    X[2][:,GW_lag,:])
            
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = X[2][:,GW_lag,:]
            
            attn_mask_SAGW = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, GW_values_DD.shape[1], 1)) # (N*heads, L, S) L: target seq len
            attn_mask_GW_lags_Module = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            
            GW_values_DD = self.SAGW_Module(K = GW_values_DD,
                                            V = GW_values_DD, 
                                            Q = GW_values_DD,
                                            attn_mask = attn_mask_SAGW)
            
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask_GW_lags_Module
                                         ))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        # Joint modules
        Output = torch.cat([GW_out.flatten(-2,-1),
                                  Weather_out.flatten(-2,-1)], dim = -1)
        
        Output = self.Linear(Output)
        
        for i in range(self.joint_mod_blocks):
            
                Output = getattr(self, f"Joint_Module_{i}")(Output,
                                                        mc_dropout = self.training or mc_dropout)
        
        # Skip connection with last observation
        if self.dropout > 0:
            Output = nn.functional.dropout1d(Output.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
                
            Output = Output.permute((0,2,1))
            
        Output = self.Output(Output)
        
        return Output.squeeze()
    
    
class ST_MultiPoint_Net_OnlyLag(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                joint_mod_blocks = 1,
                joint_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
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
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
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
                                            LayerNorm = True)

        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            LayerNorm = True)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=self.densification_dropout_dv,
                                                           p = self.densification_dropout_p)
        
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                elementwise_affine = True)
        
        ### Joint Modules #####
        
        self.Linear = nn.Sequential(nn.Linear(int(self.embedding_dim*(self.GW_W_temp_dim[0])),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        for i in range(self.joint_mod_blocks):
            setattr(self, f"Joint_Module_{i}",
                        MHA_Block(self.embedding_dim,
                                self.joint_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = 0))
            
        ### Output Layers #####
        
        self.Output = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim//2,
                                              bias = False),
                                    self.activation_fn,
                                    nn.Linear(self.embedding_dim//2, 1, bias = False))        
        
    def forward(self, X, W, Z, mc_dropout = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
                
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    X[2][:,GW_lag,:])
            
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = X[2][:,GW_lag,:]
            
            attn_mask = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        GW_out = torch.stack(GW_out, dim = -1)
        
        # Joint modules
        Output =GW_out.flatten(-2,-1)
        
        Output = self.Linear(Output)
        
        for i in range(self.joint_mod_blocks):
            
                Output = getattr(self, f"Joint_Module_{i}")(Output,
                                                        mc_dropout = self.training or mc_dropout)
        
        # Skip connection with last observation
        if self.dropout > 0:
            Output = nn.functional.dropout1d(Output.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
                
            Output = Output.permute((0,2,1))
            
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
        
        ## GW
        self.Linear_GW = nn.Sequential(nn.Linear(int(self.embedding_dim*(self.GW_W_temp_dim[0])),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        ## Source/Sink 
        self.Linear_S = nn.Sequential(nn.Linear(int(self.embedding_dim*(self.GW_W_temp_dim[1])),
                                                self.embedding_dim),
                                      self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_GW_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = self.dropout))
            
            setattr(self, f"Displacement_Module_S_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                elementwise_affine = True,
                                dropout_p = self.dropout))
        
        ### Output Layers #####
        
        self.Linear_Lag = nn.Sequential(self.activation_fn,
                                        nn.Linear(self.embedding_dim, 1))
        self.Linear_2_GW = nn.Sequential(self.activation_fn,
                                         nn.Linear(self.embedding_dim, 1))
        self.Linear_2_S = nn.Sequential(self.activation_fn,
                                        nn.Linear(self.embedding_dim, 1))  
        
        # self.Output = nn.Sequential(self.activation_fn,
        #                                 nn.Linear(1, 1))      
        
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
        
        ### Displacement modules
        
        Displacement_GW = GW_out.flatten(-2,-1)
        Displacement_GW = self.Linear_GW(Displacement_GW)
        
        Displacement_S =  Weather_out.flatten(-2,-1)
        Displacement_S = self.Linear_S(Displacement_S)
        
        for i in range(self.displacement_mod_blocks):
            
                Displacement_GW = getattr(self, f"Displacement_Module_GW_{i}")(Displacement_GW,
                                                                         mc_dropout = self.training or mc_dropout)
                
                Displacement_S = getattr(self, f"Displacement_Module_S_{i}")(Displacement_S,
                                                                         mc_dropout = self.training or mc_dropout)
        
        
        # Squeeze channel dim
        
        GW_out = self.Linear_Lag(GW_out[:,:,:,-1])
        Displacement_GW = self.Linear_2_GW(Displacement_GW)
        Displacement_S = self.Linear_2_S(Displacement_S)
        
        GW_out += Displacement_GW + Displacement_S
        
        # GW_out = self.Output(GW_out)
        
        
        return GW_out.squeeze()


class ST_MultiPoint_DisNet_K(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                emb_W = "3DCNN",
                s_coords_dim = 3,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
                dropout = 0.2, 
                activation = "GELU",
                normalization = "layernorm"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.s_coords_dim = s_coords_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.displacement_mod_blocks = displacement_mod_blocks
        self.displacement_mod_heads = displacement_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.dropout = dropout
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
        self.activation = activation
        self.emb_W = emb_W
        self.normalization = normalization
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        if self.emb_W == "3DCNN":
            self.Value_Embedding_Weather = Conv3d_Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        elif self.emb_W == "linear":
            self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = None)
            
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=self.densification_dropout_dv,
                                                           p = self.densification_dropout_p)
        
        
        ### HydrConductivity Module ####
        
        self.HydrConductivity = Embedding(in_channels = self.s_coords_dim,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = None)
        
        self.HydrConductivity_Linear = nn.Sequential(nn.Linear(self.embedding_dim, 1, bias=True),
                                                     nn.Softplus())
        
        ### Spatial Modules #####
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        
        ### Displacement Modules #####
        
        ## GW
        
        self.Linear_CD_GW = nn.Sequential(nn.Linear(self.GW_W_temp_dim[0],
                                                1),
                                    #nn.LayerNorm(1),
                                    self.activation_fn)
        
        # self.Linear_GW = nn.Sequential(nn.Linear(int(self.embedding_dim*self.GW_W_temp_dim[0]),
        #                                         self.embedding_dim),
        #                             nn.LayerNorm(self.embedding_dim),
        #                             self.activation_fn)
        
        ## Source/Sink 
        
        if self.GW_W_temp_dim[1]>0:
            self.Linear_CD_S = nn.Sequential(nn.Linear(self.GW_W_temp_dim[1],
                                                    1),
                                        #nn.LayerNorm(1),
                                        self.activation_fn)
        
        # self.Linear_S = nn.Sequential(nn.Linear(int(self.embedding_dim*self.GW_W_temp_dim[1]),
        #                                         self.embedding_dim),
        #                             nn.LayerNorm(self.embedding_dim),
        #                             self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_GW_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                normalization = self.normalization,
                                elementwise_affine = True,
                                dropout_p = 0))
            
            # setattr(self, f"Displacement_Module_S_{i}",
            #             MHA_Block(self.embedding_dim,
            #                     self.displacement_mod_heads,
            #                     self.activation,
            #                     elementwise_affine = True,
            #                     dropout_p = 0))
        
        ### Output Layers #####
        ### Linear_Lag
        Linear_Lag_list = [nn.Linear(self.embedding_dim,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_Lag_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_Lag_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                   elementwise_affine=True))
        
        Linear_Lag_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_Lag = nn.Sequential(*Linear_Lag_list)
        
        ### Linear_2_GW
        Linear_2_GW_list = [nn.Linear(self.embedding_dim,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_2_GW_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_2_GW_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                   elementwise_affine=True))
        
        Linear_2_GW_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_2_GW = nn.Sequential(*Linear_2_GW_list)
        
        ### Linear_2_S
        Linear_2_S_list = [nn.Linear(self.embedding_dim,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_2_S_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_2_S_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                   elementwise_affine=True))
        
        Linear_2_S_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_2_S = nn.Sequential(*Linear_2_S_list)
        
        # self.Output = nn.Sequential(nn.Linear(3,1),
        #                             self.activation_fn,
        #                             nn.Linear(1,1))
         
        
    def forward(self, X, W, Z, mc_dropout = False, get_displacement_terms = False, get_lag_term = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        HydrConductivity = self.HydrConductivity(Z[:,:,:self.s_coords_dim])
        HydrConductivity = self.HydrConductivity_Linear(HydrConductivity)
        
        if self.emb_W == "3DCNN":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input).permute((0,2,3,4,1)).flatten(2,3)
        elif self.emb_W == "linear":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input)
            
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    X[2][:,GW_lag,:])
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = X[2][:,GW_lag,:]
            
            attn_mask = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        GW_out = torch.stack(GW_out, dim = -1)
        Weather_out = torch.stack(Weather_out, dim = -1)
        
        ### Displacement modules
        
        #Displacement_GW = GW_out.flatten(-2,-1)
        Displacement_GW = self.Linear_CD_GW(GW_out).flatten(-2,-1)
        Displacement_GW_skip = Displacement_GW
        
        Displacement_S = Weather_out#.flatten(-2,-1)
        if self.GW_W_temp_dim[1]>0:
            Displacement_S = self.Linear_CD_S(Displacement_S)
            
        Displacement_S = Displacement_S.flatten(-2,-1)
        
        if self.dropout > 0:
            Displacement_GW = nn.functional.dropout1d(Displacement_GW.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
            
            Displacement_S = nn.functional.dropout1d(Displacement_S.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
            
            Displacement_GW = Displacement_GW.permute((0,2,1))
            Displacement_S = Displacement_S.permute((0,2,1))
        
        for i in range(self.displacement_mod_blocks):
            
                Displacement_GW = getattr(self, f"Displacement_Module_GW_{i}")(Displacement_GW,
                                                                         mc_dropout = self.training or mc_dropout)
                
                # Displacement_S = getattr(self, f"Displacement_Module_S_{i}")(Displacement_S,
                #                                                          mc_dropout = self.training or mc_dropout)
        
        
        ## Squeeze channel dim
        
        #GW_lag_out = self.Linear_Lag(GW_out.permute((0,3,1,2))) # N, D, S,  C, 
        
        GW_lag_out = self.Linear_Lag(Displacement_GW_skip)
        
        # GW Continuity equation estimation
        Displacement_GW = self.Linear_2_GW(Displacement_GW) # Darcy velocity
        Displacement_GW = HydrConductivity * Displacement_GW # Weight by HydrConductivity: ISOTROPIC Conductivity Field
        
        Displacement_S = self.Linear_2_S(Displacement_S)
        
        #SUM
        Y_hat = GW_lag_out + Displacement_GW + Displacement_S # Euler method
        
        #Concat
        # Y_hat = torch.cat([GW_lag_out[:,-1,:,:],
        #                    Displacement_GW,
        #                    Displacement_S],
        #                   dim = -1) 
        
        # Y_hat = self.Output(Y_hat)
        
        if get_displacement_terms is False:
            
            return Y_hat.squeeze()
        
        else: 
            
            output_list = [Y_hat.squeeze(),
                        Displacement_GW.squeeze(),
                        Displacement_S.squeeze(),
                        HydrConductivity.squeeze()]
            if get_lag_term is False:
            
                return output_list
                
            else:
            
                output_list.append(GW_lag_out.squeeze())
                return output_list
            
            
class ST_MultiPoint_DisNet_SAGW_K(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                emb_W = "3DCNN",
                s_coords_dim = 3,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
                dropout = 0.2, 
                activation = "GELU",
                normalization = "layernorm"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.s_coords_dim = s_coords_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.displacement_mod_blocks = displacement_mod_blocks
        self.displacement_mod_heads = displacement_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.dropout = dropout
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
        self.activation = activation
        self.emb_W = emb_W
        self.normalization = normalization
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None,
                                            elementwise_affine = False)
        if self.emb_W == "3DCNN":
            self.Value_Embedding_Weather = Conv3d_Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None,
                                                elementwise_affine = False,
                                                pooling = True)
        elif self.emb_W == "linear":
            self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = None,
                                    elementwise_affine = False)
            
        
        self.S_coords_Embedding = Embedding(in_channels = self.s_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None,
                                            elementwise_affine = False)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=0,
                                                           p = self.densification_dropout_p)
        
        
        ### HydrConductivity Module ####
        
        self.HydrConductivity = Embedding(in_channels = self.s_coords_dim,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = self.normalization,
                                    norm_dims="1d",
                                    elementwise_affine = False)
        
        self.HydrConductivity_Linear = nn.Sequential(nn.Linear(self.embedding_dim, 1, bias=True),
                                                     nn.Softplus())
        
        ### Spatial Modules #####
        self.SAGW_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = False)
        
        
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = False)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = False)
        
        
        ### Displacement Modules #####
        
        ## GW
        self.Linear_CD_GW = nn.Sequential(nn.Linear(self.GW_W_temp_dim[0],
                                                1),
                                    self.activation_fn)
        
        ## Source/Sink 
        if self.GW_W_temp_dim[1]>0:
            self.Linear_CD_S = nn.Sequential(nn.Linear(self.GW_W_temp_dim[1],
                                                    1),
                                        self.activation_fn)
        
        self.Linear_1_GW =  nn.Sequential(nn.Linear(self.embedding_dim+2,
                                self.embedding_dim),
                                self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_GW_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                normalization = self.normalization,
                                elementwise_affine = False,
                                dropout_p = 0))
        
        
        ### Output Layers #####
        
        Linear_Lag_list = [nn.Linear(self.embedding_dim+2,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_Lag_list.append(nn.LayerNorm(self.embedding_dim//2,
                                    elementwise_affine=False))
            elif self.normalization == "instancenorm":
                Linear_Lag_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                   elementwise_affine=False))
        
        Linear_Lag_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_Lag = nn.Sequential(*Linear_Lag_list)
        
        ### Linear_2_GW
        Linear_2_GW_list = [nn.Linear(self.embedding_dim,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_2_GW_list.append(nn.LayerNorm(self.embedding_dim//2,
                                elementwise_affine=False))
            elif self.normalization == "instancenorm":
                Linear_2_GW_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                elementwise_affine=False))
        
        Linear_2_GW_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_2_GW = nn.Sequential(*Linear_2_GW_list)
        
        ### Linear_2_S
        Linear_2_S_list = [nn.Linear(self.embedding_dim+2,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_2_S_list.append(nn.LayerNorm(self.embedding_dim//2,
                                    elementwise_affine=False))
            elif self.normalization == "instancenorm":
                Linear_2_S_list.append(InstanceNorm1d_MA(self.embedding_dim//2,1,-1,
                                   elementwise_affine=False))
        
        Linear_2_S_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_2_S = nn.Sequential(*Linear_2_S_list)
         
        
    def forward(self, X, W, Z, mc_dropout = False, get_displacement_terms = False, get_lag_term = False):
        
        ### Embedding #####
        
        #GW_temp_means = torch.mean(X[0], dim = 1)
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1],
                               X[2].unsqueeze(-1)], dim = -1)) #N, D, S, C
        
        GW_values_mask = X[2] * False
        
        GW_s_coords = self.S_coords_Embedding(X[1][:,:,:,:self.s_coords_dim]) #N, D, S, C
        
        Z_s_coords = self.S_coords_Embedding(Z[:,:,:self.s_coords_dim].unsqueeze(2))[:,:,0,:]
        
        HydrConductivity = self.HydrConductivity(Z[:,:,:self.s_coords_dim])
        HydrConductivity = self.HydrConductivity_Linear(HydrConductivity)
        
        if self.emb_W == "3DCNN":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input).permute((0,2,3,4,1)).flatten(2,3)
        elif self.emb_W == "linear":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input)
            
        Weather_s_coords = self.S_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)[:,:,:,:self.s_coords_dim]) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    GW_values_mask[:,GW_lag,:])
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = GW_values_mask[:,GW_lag,:]
            
            attn_mask_SAGW = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, GW_values_DD.shape[1], 1)) # (N*heads, L, S) L: target seq len
            attn_mask_GW_lags_Module = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_s_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            
            GW_values_DD = self.SAGW_Module(K = GW_values_DD,
                                            V = GW_values_DD, 
                                            Q = GW_values_DD,
                                            attn_mask = attn_mask_SAGW)
            
            GW_values_DD += GW_values[:,GW_lag,:,:]
            
            GW_out.append(self.GW_lags_Module(K = GW_s_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_s_coords,
                                         attn_mask = attn_mask_GW_lags_Module
                                         ))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_s_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_s_coords))
        
        X_temp_enc = X[1][:,:,0,self.s_coords_dim:][:,:,None,:].permute(0,2,3,1)
        GW_out = torch.stack(GW_out, dim = -1)
        GW_out = torch.cat([GW_out,
                            X_temp_enc.repeat(1,GW_out.shape[1],1,1)],
                           dim = -2) 
        
        #self.cache_GW_out = GW_out
        W_temp_enc = W[1][:,self.s_coords_dim:,:,0,0][:,:,:,None].permute(0,3,1,2)
        Weather_out = torch.stack(Weather_out, dim = -1)
        Weather_out = torch.cat([Weather_out,
                            W_temp_enc.repeat(1,Weather_out.shape[1],1,1)],
                            dim = -2) 
        
        ### Displacement modules
        
        #Displacement_GW = GW_out.flatten(-2,-1)
        Displacement_GW = self.Linear_CD_GW(GW_out).flatten(-2,-1)
        Displacement_GW_skip = Displacement_GW
        
        #Displacement_S = Weather_out#.flatten(-2,-1)
        Displacement_S = self.Linear_CD_S(Weather_out).flatten(-2,-1)
        
        if self.dropout > 0:
            Displacement_GW = nn.functional.dropout1d(Displacement_GW.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
            
            Displacement_S = nn.functional.dropout1d(Displacement_S.permute((0,2,1)),
                                                p = self.dropout, training = self.training or mc_dropout)
            
            Displacement_GW = Displacement_GW.permute((0,2,1))
            Displacement_S = Displacement_S.permute((0,2,1))
        
        Displacement_GW = self.Linear_1_GW(Displacement_GW)
        
        for i in range(self.displacement_mod_blocks):
            
                Displacement_GW = getattr(self, f"Displacement_Module_GW_{i}")(Displacement_GW,
                                                                         mc_dropout = self.training or mc_dropout)
                
                # Displacement_S = getattr(self, f"Displacement_Module_S_{i}")(Displacement_S,
                #                                                          mc_dropout = self.training or mc_dropout)
        
        
        ## Squeeze channel dim
        
        #GW_lag_out = self.Linear_Lag(GW_out.permute((0,3,1,2))) # N, D, S,  C, 
        
        GW_lag_out = self.Linear_Lag(Displacement_GW_skip)
        #GW_lag_out += X[0][0,:].unsqueeze(-1)
        
        # GW Continuity equation estimation
        Displacement_GW = self.Linear_2_GW(Displacement_GW) # Darcy velocity
        Displacement_GW = HydrConductivity * Displacement_GW # Weight by HydrConductivity: ISOTROPIC Conductivity Field
        
        Displacement_S = self.Linear_2_S(Displacement_S)
        
        #SUM
        Y_hat = GW_lag_out + Displacement_GW + Displacement_S # Euler method
        
        #Concat
        # Y_hat = torch.cat([GW_lag_out[:,-1,:,:],
        #                    Displacement_GW,
        #                    Displacement_S],
        #                   dim = -1) 
        
        # Y_hat = self.Output(Y_hat)
        
        if get_displacement_terms is False:
            
            return Y_hat.squeeze()
        
        else: 
            
            output_list = [Y_hat.squeeze(),
                        Displacement_GW.squeeze(),
                        Displacement_S.squeeze(),
                        HydrConductivity.squeeze()]
            if get_lag_term is False:
            
                return output_list
                
            else:
            
                output_list.append(GW_lag_out.squeeze())
                return output_list
            
class ST_MultiPoint_DisNet_SAGW_K_ALT(nn.Module):
    
    def __init__(self,
                value_dim_GW = 6,
                value_dim_Weather = 9, 
                embedding_dim = 16,
                emb_W = "3DCNN",
                s_coords_dim = 3,
                st_coords_dim = 5,
                spatial_mha_heads = 2,
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [2,5],
                densification_dropout_p = 0.25,
                densification_dropout_dv = 0,
                dropout = 0.2, 
                activation = "GELU",
                normalization = "layernorm"):
        
        super().__init__()
        
        self.value_dim_GW = value_dim_GW
        self.value_dim_Weather = value_dim_Weather
        self.embedding_dim = embedding_dim
        self.s_coords_dim = s_coords_dim
        self.st_coords_dim = st_coords_dim
        self.spatial_mha_heads = spatial_mha_heads
        self.displacement_mod_blocks = displacement_mod_blocks
        self.displacement_mod_heads = displacement_mod_heads
        self.GW_W_temp_dim = GW_W_temp_dim
        self.dropout = dropout
        self.densification_dropout_p = densification_dropout_p
        self.densification_dropout_dv = densification_dropout_dv
        self.activation = activation
        self.emb_W = emb_W
        self.normalization = normalization
        
        if self.activation == "LeakyReLU":
            self.activation_fn = nn.LeakyReLU()
        elif self.activation == "GELU":
            self.activation_fn = nn.GELU()
        
        ### Embedding #####
        self.Value_Embedding_GW = Embedding(in_channels = self.value_dim_GW,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        if self.emb_W == "3DCNN":
            self.Value_Embedding_Weather = Conv3d_Embedding(in_channels = self.value_dim_Weather,
                                                hidden_channels = self.embedding_dim,
                                                out_channels= self.embedding_dim,
                                                activation = self.activation,
                                                normalization = None)
        elif self.emb_W == "linear":
            self.Value_Embedding_Weather = Embedding(in_channels = self.value_dim_Weather,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = None)
            
        
        self.ST_coords_Embedding = Embedding(in_channels = self.st_coords_dim,
                                            hidden_channels = self.embedding_dim,
                                            out_channels= self.embedding_dim,
                                            activation = self.activation,
                                            normalization = None)
        
        ### Densification Dropout 
        
        self.Densification_Dropout = Densification_Dropout(drop_value=self.densification_dropout_dv,
                                                           p = self.densification_dropout_p)
        
        
        ### HydrConductivity Module ####
        
        self.HydrConductivity = Embedding(in_channels = self.s_coords_dim,
                                    hidden_channels = self.embedding_dim,
                                    out_channels= self.embedding_dim,
                                    activation = self.activation,
                                    normalization = self.normalization)
        
        self.HydrConductivity_Linear = nn.Sequential(nn.Linear(self.embedding_dim, 1, bias=True),
                                                     nn.Softplus())
        
        ### Spatial Modules #####
        self.SAGW_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        
        self.GW_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        self.Weather_lags_Module = Spatial_MHA_Block(embedding_dim = self.embedding_dim,
                                                heads = self.spatial_mha_heads,
                                                output_channels = self.embedding_dim,
                                                activation = self.activation,
                                                normalization = self.normalization,
                                                elementwise_affine = True)
        
        
        ### Displacement Modules #####
        
        ## GW
        
        # self.Linear_CD_GW = nn.Sequential(nn.Linear(self.GW_W_temp_dim[0],
        #                                         1),
        #                             #nn.LayerNorm(1),
        #                             self.activation_fn)
        
        Linear_Squeeze_GW_list = [nn.Linear((self.embedding_dim+5)*self.GW_W_temp_dim[0],
                                    self.embedding_dim)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_Squeeze_GW_list.append(nn.LayerNorm(self.embedding_dim))
            elif self.normalization == "instancenorm":
                Linear_Squeeze_GW_list.append(InstanceNorm1d_MA(self.embedding_dim,1,-1,
                                   elementwise_affine=True))
        Linear_Squeeze_GW_list.append(self.activation_fn)
        
        
        
        self.Linear_Squeeze_GW = nn.Sequential(*Linear_Squeeze_GW_list)
        
        ## Source/Sink 
        
        # if self.GW_W_temp_dim[1]>0:
        #     self.Linear_CD_S = nn.Sequential(nn.Linear(self.GW_W_temp_dim[1],
        #                                             1),
        #                                 #nn.LayerNorm(1),
        #                                 self.activation_fn)
        
        # self.Linear_S = nn.Sequential(nn.Linear(int(self.embedding_dim*self.GW_W_temp_dim[1]),
        #                                         self.embedding_dim),
        #                             nn.LayerNorm(self.embedding_dim),
        #                             self.activation_fn)
        
        for i in range(self.displacement_mod_blocks):
            setattr(self, f"Displacement_Module_GW_{i}",
                        MHA_Block(self.embedding_dim,
                                self.displacement_mod_heads,
                                self.activation,
                                normalization = self.normalization,
                                elementwise_affine = True,
                                dropout_p = 0))
            
            # setattr(self, f"Displacement_Module_S_{i}",
            #             MHA_Block(self.embedding_dim,
            #                     self.displacement_mod_heads,
            #                     self.activation,
            #                     elementwise_affine = True,
            #                     dropout_p = 0))
        
        ### Output Layers #####
        
        ### Linear_TD_Lag
        Linear_TD_Lag_list = [nn.Linear(self.embedding_dim+5,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_TD_Lag_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_TD_Lag_list.append(InstanceNorm2d_MA(self.embedding_dim//2,-1,1,
                                   elementwise_affine=True))
        
        Linear_TD_Lag_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_TD_Lag = nn.Sequential(*Linear_TD_Lag_list)
        
        ### Linear_TD_GW
        Linear_TD_GW_list = [nn.Linear(self.embedding_dim,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_TD_GW_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_TD_GW_list.append(InstanceNorm1d_MA(self.embedding_dim//2,-1,1,
                                   elementwise_affine=True))
        
        Linear_TD_GW_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_TD_GW = nn.Sequential(*Linear_TD_GW_list)
        
        ### Linear_TD_S
        Linear_TD_S_list = [nn.Linear(self.embedding_dim+5,
                                    self.embedding_dim//2)]
        if self.normalization is not None:
            if self.normalization == "layernorm":
                Linear_TD_S_list.append(nn.LayerNorm(self.embedding_dim//2))
            elif self.normalization == "instancenorm":
                Linear_TD_S_list.append(InstanceNorm2d_MA(self.embedding_dim//2,-1,1,
                                   elementwise_affine=True))
        
        Linear_TD_S_list.extend([self.activation_fn,
                                nn.Linear(self.embedding_dim//2, 1,
                                                bias=True)])
        
        self.Linear_TD_S = nn.Sequential(*Linear_TD_S_list)
        
        # self.Output = nn.Sequential(nn.Linear(3,1),
        #                             self.activation_fn,
        #                             nn.Linear(1,1))
         
        
    def forward(self, X, W, Z, mc_dropout = False, get_displacement_terms = False, get_lag_term = False):
        
        ### Embedding #####
        
        GW_values = self.Value_Embedding_GW(torch.cat([X[0].unsqueeze(-1),
                               X[1]], dim = -1)) #N, D, S, C
        
        GW_st_coords = self.ST_coords_Embedding(X[1]) #N, D, S, C
        
        Z_st_coords = self.ST_coords_Embedding(Z)
        
        HydrConductivity = self.HydrConductivity(Z[:,:,:self.s_coords_dim])
        HydrConductivity = self.HydrConductivity_Linear(HydrConductivity)
        
        if self.emb_W == "3DCNN":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input).permute((0,2,3,4,1)).flatten(2,3)
        elif self.emb_W == "linear":
            Weather_values_input = torch.cat([W[0], W[1]], dim = 1).permute((0,2,3,4,1)).flatten(2,3)
            Weather_values = self.Value_Embedding_Weather(Weather_values_input)
            
        Weather_st_coords = self.ST_coords_Embedding(W[1].permute((0,2,3,4,1)).flatten(2,3)) #N, D, S, C
        
        GW_out = []
        Weather_out = []
        
        for GW_lag in range(GW_values.shape[1]):
            
            if self.training or mc_dropout:
                GW_values_DD, GW_mask_DD = self.Densification_Dropout(GW_values[:,GW_lag,:,:],
                                                                    X[2][:,GW_lag,:])
            else: 
                GW_values_DD = GW_values[:,GW_lag,:,:]
                GW_mask_DD = X[2][:,GW_lag,:]
            
            attn_mask_SAGW = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, GW_values_DD.shape[1], 1)) # (N*heads, L, S) L: target seq len
            attn_mask_GW_lags_Module = GW_mask_DD[:,None,:].repeat((self.spatial_mha_heads, Z_st_coords.shape[1], 1)) # (N*heads, L, S) L: target seq len
            
            GW_values_DD = self.SAGW_Module(K = GW_values_DD,
                                            V = GW_values_DD, 
                                            Q = GW_values_DD,
                                            attn_mask = attn_mask_SAGW)
            
            GW_out.append(self.GW_lags_Module(K = GW_st_coords[:,GW_lag,:,:],
                                         V = GW_values_DD,
                                         Q = Z_st_coords,
                                         attn_mask = attn_mask_GW_lags_Module
                                         ))
            
        for Weather_lag in range(Weather_values.shape[1]):
            Weather_out.append(self.Weather_lags_Module(K = Weather_st_coords[:,Weather_lag,:,:],
                                                        V = Weather_values[:,Weather_lag,:,:],
                                                        Q = Z_st_coords))
        
        
        GW_out = torch.stack(GW_out, dim = -2)
        #self.cache_GW_out = GW_out
        Weather_out = torch.stack(Weather_out, dim = -2)
        
        GW_out = torch.cat([GW_out,
                            Z[:,:,None,:].expand(-1,-1,self.GW_W_temp_dim[0],-1)],
                           dim = -1) 
        Weather_out = torch.cat([Weather_out,
                            Z[:,:,None,:].expand(-1,-1,self.GW_W_temp_dim[1],-1)],
                           dim = -1) 
        ## TD Layers
        GW_lag_out = torch.mean(self.Linear_TD_Lag(GW_out), dim = -2)
        Displacement_GW = self.Linear_Squeeze_GW(GW_out.flatten(-2,-1))
        Displacement_S = torch.mean(self.Linear_TD_S(Weather_out), dim = -2)
        
        ### Displacement modules
          
        for i in range(self.displacement_mod_blocks):
            
                Displacement_GW = getattr(self, f"Displacement_Module_GW_{i}")(Displacement_GW,
                                                                         mc_dropout = self.training or mc_dropout)

        # GW Continuity equation estimation
        Displacement_GW = self.Linear_TD_GW(Displacement_GW) # Darcy velocity
        Displacement_GW = HydrConductivity * Displacement_GW # Weight by HydrConductivity: ISOTROPIC Conductivity Field
        
        
        
        #SUM
        Y_hat = GW_lag_out + Displacement_GW + Displacement_S # Euler method
                
        if get_displacement_terms is False:
            
            return Y_hat.squeeze()
        
        else: 
            
            output_list = [Y_hat.squeeze(),
                        Displacement_GW.squeeze(),
                        Displacement_S.squeeze(),
                        HydrConductivity.squeeze()]
            if get_lag_term is False:
            
                return output_list
                
            else:
            
                output_list.append(GW_lag_out.squeeze())
                return output_list
    
    
class ST_MultiPoint_DisNet_alt(nn.Module):
    
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
    