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
            
            target_Icond = torch.moveaxis(target_Icond, -1, 1)
            target_Icond = torch.reshape(target_Icond, (*target_Icond.shape[:2],
                                         Z.shape[1], Z.shape[2]))

            return target_Icond

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
        
        
        # self.norm_layer_1 = nn.LayerNorm(normalized_shape = embedding_dim, 
        #                                  elementwise_affine = self.elementwise_affine)
        
        self.mha = nn.MultiheadAttention(embedding_dim, heads,
                                         batch_first=True)
        
        #self.dropout_1 = nn.Dropout1d(self.dropout_p)
        
        # self.norm_layer_2 = nn.LayerNorm(normalized_shape = embedding_dim,
        #                                  elementwise_affine = self.elementwise_affine)
        
        
        self.mlp = nn.Sequential(
                                    nn.Linear(embedding_dim, embedding_dim),
                                    self.activation,
                                    nn.Linear(embedding_dim, embedding_dim),
                                    )
        
        #self.dropout_2 = nn.Dropout1d(self.dropout_p)
        
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
        
        # output = nn.functional.dropout1d(output.permute((0,2,1)),
        #                                  p = self.dropout_p, training = self.training or mc_dropout)
        
        # output = output.permute((0,2,1))
        
        output = output + skip_1
        skip_2 = output #.clone()
        
        #output = self.norm_layer_2(output)
        output = self.mlp(output)
        
        output = nn.functional.dropout1d(output.permute((0,2,1)),
                                         p = self.dropout_p, training = self.training or mc_dropout)
        
        output = output.permute((0,2,1))
        
        output = output + skip_2
        
        output = self.activation(output)
        
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
    
class Topographical_Embdedding(nn.Module):
    
    def __init__(self, embedding_dim, activation):
        super().__init__()
        
        self.activation = activation
        
        topo_embeddings = []
        topo_embeddings.append(nn.Linear(3, embedding_dim)) #(3: lat, lon, height)
        topo_embeddings.append(self.activation)
        self.topo_embeddings = nn.Sequential(*topo_embeddings)
        
    def forward(self, input):
        
        return self.topo_embeddings(input)
        
class Spatial_Attention_Block(nn.Module):
    
    def __init__(self,
                 embedding_dim,
                 input_channles,
                 heads,
                 output_channels,
                 activation,
                 elementwise_affine,
                 Topo_Embedder = None):
        super().__init__()
        
        self.elementwise_affine = elementwise_affine
        
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
        
        self.norm_linear = nn.Sequential(
                                    #nn.LayerNorm(normalized_shape = int(embedding_dim*2), elementwise_affine=self.elementwise_affine),
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

class SparseData_Transformer(nn.Module):
    
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
                                                            hidden_channels = 32, 
                                                            out_channels = 1, 
                                                            LayerNorm = False,
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
        
        self.linear = nn.Sequential(nn.Linear(self.fusion_embedding_dim,
                                        self.spatial_embedding_dim),
                                    self.activation_fn)
                                   
        self.output = nn.Sequential(nn.Linear(self.spatial_embedding_dim,
                                            self.spatial_embedding_dim),
                                    #nn.LayerNorm(self.spatial_embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.spatial_embedding_dim, 1))
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False):
        
        ### Weather module ### 
        
        ST_conditioning_input = W[1][:,:,None,:].expand(-1,-1,Z.shape[1],-1)
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z[:,None,:,:].expand(-1,W[1].shape[1],-1,-1)],
                                          dim = -1) # B, D, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) # N,D,S,2*FiLMed,C # ST N,D,S,C_out
        ST_Conditionings = torch.flatten(ST_Conditionings, 1, 2)
        
        if [W[0].shape[2], Z.shape[1]] != self.target_dim:
                
                print("Computing Causal-Mask...")
            
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
            
            Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
                                                p = self.spatial_dropout, training = True)
            
            Fused_st = self.Fusion_Embedding(torch.moveaxis(Fused_st, 1, -1))
            Fused_st += ST_Conditionings
            
            for i in range(self.st_mha_blocks):
            
                Fused_st = getattr(self, f"MHA_Block_{i}")(Fused_st, causal_masks, self.training)
            
            Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1])) 
            
            Fused_st = self.linear(Fused_st)
            
            Fused_st = Fused_st + Autoreg_st
            
            Output_st = self.output(Fused_st)
            
            return Output_st.squeeze()
        
        else:
            
            Sparse_data = X
            Sparse_data_mask = X_mask
            
            Autoreg_st_rlist = []
            Weather_st_rlist = []
                
            # Unfolding time - iterate own prediction as input
            for timestep in tqdm(range(W[0].shape[2])):
                
                Sparse_data_mask = torch.repeat_interleave(Sparse_data_mask, self.spatial_heads, dim = 0)
                
                Autoreg_st_rlist.append(self.SparseAutoreg_Module(K = Sparse_data[:,:,:3],
                                                                V = Sparse_data,
                                                                Q = Z,
                                                                attn_mask = ~Sparse_data_mask[:,None,:].expand(-1,Z.shape[1],-1)))
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_st_rlist.append(self.Weather_Module(
                                                        K = Weather_Keys,
                                                        V = Weather_Values,
                                                        Q = Z
                                                        ))
                
                Autoreg_st = torch.stack(Autoreg_st_rlist, dim = 1)
                Weather_st = torch.stack(Weather_st_rlist, dim = 1)
                Fused_st = torch.cat([Autoreg_st,
                                    Weather_st], 
                                    dim = -1)
                
                Fused_st_extent = Fused_st.shape[1:3]
                Fused_st = torch.flatten(Fused_st, 1, 2)
                
                Fused_st =  nn.functional.dropout1d(Fused_st.permute((0,2,1)),
                                                p = self.spatial_dropout, training = mc_dropout)
                
                Fused_st = self.Fusion_Embedding(Fused_st.permute((0,2,1)))
                Fused_st = Fused_st + ST_Conditionings[:,:Fused_st.shape[1],:]
                
                causal_mask = causal_masks[:,
                                          :math.prod(Fused_st_extent),
                                          :math.prod(Fused_st_extent)]
                
                for i in range(self.st_mha_blocks):
            
                    Fused_st = getattr(self, f"MHA_Block_{i}")(Fused_st,
                                                                   causal_mask,
                                                                   mc_dropout)
                
                
                Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1]))        
                
                Fused_st = self.linear(Fused_st)
                Fused_st = Fused_st + Autoreg_st
            
                Output_st = self.output(Fused_st)
            
                Sparse_data = torch.cat([Z.clone(),
                                       Output_st[:,-1,:,:]],
                                      dim=-1)

                Sparse_data_mask = torch.ones((Sparse_data[:,:,0].shape)).to(torch.bool).to(Sparse_data.device)
            
            Output_st = Output_st.squeeze()
            
            return Output_st

class SparseData_Transformer_MoE(nn.Module):
    
    def __init__(self,
                weather_CHW_dim = [7, 9, 12],
                target_dim = [14, 31],
                spatial_embedding_dim = 16,
                spatial_heads = 2,
                fusion_embedding_dim = 128,
                st_heads = 4,
                num_experts = 8,
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
        self.num_experts = num_experts
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
        
        self.topo_embedder = Topographical_Embdedding(embedding_dim = self.spatial_embedding_dim,
                                                      activation = self.activation_fn)
        
        self.SparseAutoreg_Module = Spatial_Attention_Block_MoE(
                                embedding_dim = self.spatial_embedding_dim,
                                input_channles = 4,
                                heads = self.spatial_heads,
                                num_experts= self.num_experts,
                                output_channels = self.spatial_embedding_dim,
                                activation = self.activation,
                                elementwise_affine= self.layernorm_affine,
                                STMoE_prenorm = False,
                                Topo_Embedder = self.topo_embedder)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 6,
                                                            hidden_channels = 32, 
                                                            out_channels = 1, 
                                                            LayerNorm = False,
                                                            activation = self.activation)

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block_MoE(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 num_experts= self.num_experts,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine,
                 STMoE_prenorm = False,
                 Topo_Embedder = self.topo_embedder)
        
        ### Joint Modoule ### 
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(self.spatial_embedding_dim,
                                                       self.fusion_embedding_dim),
                                             self.activation_fn)
        
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
        
        self.linear = nn.Sequential(nn.Linear(self.fusion_embedding_dim,
                                        self.spatial_embedding_dim),
                                    self.activation_fn)
                                   
        self.output = nn.Sequential(nn.LayerNorm(self.spatial_embedding_dim),
                                    nn.Linear(self.spatial_embedding_dim,
                                            self.spatial_embedding_dim),
                                    self.activation_fn,
                                    nn.Linear(self.spatial_embedding_dim, 1))
        
    def forward(self, X, Z, W, X_mask, teacher_forcing = False, mc_dropout = False,
                get_aux_loss = False):
        
        ### Weather module ### 
        
        ST_conditioning_input = W[1][:,:,None,:].expand(-1,-1,Z.shape[1],-1).clone()
        ST_conditioning_input = torch.cat([ST_conditioning_input,
                                           Z[:,None,:,:].expand(-1,W[1].shape[1],-1,-1).clone()],
                                          dim = -1) # B, D, S, C
        
        ST_Conditionings = self.ST_Conditioning_Module(ST_conditioning_input) # N,D,S,2*FiLMed,C # ST N,D,S,C_out
        ST_Conditionings = torch.flatten(ST_Conditionings, 1, 2)
        
        if [W[0].shape[2], Z.shape[1]] != self.target_dim:
                
                print("Computing Causal-Mask...")
            
                # Causal Mask
                causal_masks = torch.tril(torch.ones(W[0].shape[2], W[0].shape[2])) # Tempooral Mask
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                            dim = 0)
                causal_masks = causal_masks.repeat_interleave(Z.shape[1],  # Repeating for spatial extent
                                                        dim = 1)
                
                causal_masks = ~causal_masks.to(torch.bool)
                causal_masks = causal_masks[None,:,:].expand(int(W[0].shape[0]*self.st_heads), -1,-1).clone().to(W[0].device)

                
        else:
                causal_masks = self.causal_mask[None,:,:].expand(int(W[0].shape[0]*self.st_heads),-1,-1).clone().to(W[0].device)
        
        if self.training is True and teacher_forcing is True:
            Autoreg_st = []
            Weather_st = []
            Autoreg_aux_loss = []
            Weather_aux_loss = []
            total_aux_loss = []
            
            for timestep in range(X.shape[1]):
                 
                if self.densification_dropout_p>0:
                    X_t, X_mask_t = densification_dropout([X[:,timestep,:,:],
                                                           X_mask[:,timestep,:]],
                                                        p = self.densification_dropout_p)
                else:
                    X_t = X[:,timestep,:,:]
                    X_mask_t = X_mask[:,timestep,:]
                
                #X_mask_t = torch.repeat_interleave(X_mask_t, self.spatial_heads, dim = 0)
                X_mask_t = ~X_mask_t[:,None,:].repeat((self.spatial_heads,Z.shape[1],1))
                
                Autoreg_st_out, Autoreg_st_aux_loss, _ ,_ = self.SparseAutoreg_Module(
                                                                    K = X_t[:,:,:3],
                                                                    V = X_t,
                                                                    Q = Z,
                                                                    attn_mask = X_mask_t
                                                                    )
                Autoreg_st.append(Autoreg_st_out)
                Autoreg_aux_loss.append(Autoreg_st_aux_loss)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                Weather_st_out, Weather_st_aux_loss, _ , _  = self.Weather_Module(
                                                                K = Weather_Keys,
                                                                V = Weather_Values,
                                                                Q = Z
                                                                )
                Weather_st.append(Weather_st_out)
                Weather_aux_loss.append(Weather_st_aux_loss)
                
            Autoreg_aux_loss = torch.tensor(Autoreg_aux_loss).mean()
            Weather_aux_loss = torch.tensor(Weather_aux_loss).mean()
            Autoreg_st = torch.stack(Autoreg_st, dim = 1)
            Weather_st = torch.stack(Weather_st, dim = 1)

            # Fused_st = torch.cat([Autoreg_st,
            #                         Weather_st], 
            #                         dim = -1)
            
            Fused_st = Autoreg_st + Weather_st
            
            Fused_st_extent = Fused_st.shape[1:3]
            Fused_st = torch.flatten(Fused_st, 1, 2)
            
            Fused_st =  nn.functional.dropout1d(torch.moveaxis(Fused_st, 1, -1),
                                                p = self.spatial_dropout, training = True)
            
            Fused_st = self.Fusion_Embedding(torch.moveaxis(Fused_st, 1, -1))
            Fused_st += ST_Conditionings
            
            for i in range(self.st_mha_blocks):
            
                Fused_st, aux_loss, _ , _ = getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st, causal_masks, self.training)
                total_aux_loss.append(aux_loss)
            
            total_aux_loss = torch.tensor(total_aux_loss).sum()
            total_aux_loss += Autoreg_aux_loss + Weather_aux_loss
            
            Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1])) 
            
            Fused_st = self.linear(Fused_st)
            
            Fused_st = Fused_st + Autoreg_st
            
            Output_st = self.output(Fused_st).squeeze()
            
            if get_aux_loss is True:
                return [Output_st, total_aux_loss]
            else:
                return Output_st
        
        else:
            
            Sparse_data = X
            Sparse_data_mask = X_mask
            
            Autoreg_st_rlist = []
            Weather_st_rlist = []
            Autoreg_aux_loss = []
            Weather_aux_loss = []
            total_aux_loss = []
                
            # Unfolding time - iterate own prediction as input
            for timestep in tqdm(range(W[0].shape[2])):
                
                #Sparse_data_mask = torch.repeat_interleave(Sparse_data_mask, self.spatial_heads, dim = 0)
                Sparse_data_mask = ~Sparse_data_mask[:,None,:].repeat((self.spatial_heads, Z.shape[1], 1))
                
                Autoreg_st_out, Autoreg_st_aux_loss, _ ,_ = self.SparseAutoreg_Module(K = Sparse_data[:,:,:3],
                                                                V = Sparse_data,
                                                                Q = Z,
                                                                attn_mask = Sparse_data_mask)
                Autoreg_st_rlist.append(Autoreg_st_out)
                Autoreg_aux_loss.append(Autoreg_st_aux_loss)
                
                Weather_Keys = torch.moveaxis(W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1) # 
                Weather_Values = torch.moveaxis(W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1), 1, -1)
                
                Weather_st_out, Weather_st_aux_loss, _ , _ = self.Weather_Module(
                                                        K = Weather_Keys,
                                                        V = Weather_Values,
                                                        Q = Z
                                                        )
                Weather_st_rlist.append(Weather_st_out)
                Weather_aux_loss.append(Weather_st_aux_loss)
                
                Autoreg_st = torch.stack(Autoreg_st_rlist, dim = 1)
                Weather_st = torch.stack(Weather_st_rlist, dim = 1)
                # Fused_st = torch.cat([Autoreg_st,
                #                     Weather_st], 
                #                     dim = -1)
                
                Fused_st = Autoreg_st + Weather_st
                
                Fused_st_extent = Fused_st.shape[1:3]
                Fused_st = torch.flatten(Fused_st, 1, 2)
                
                Fused_st =  nn.functional.dropout1d(Fused_st.permute((0,2,1)),
                                                p = self.spatial_dropout, training = mc_dropout)
                
                Fused_st = self.Fusion_Embedding(Fused_st.permute((0,2,1)))
                Fused_st = Fused_st + ST_Conditionings[:,:Fused_st.shape[1],:]
                
                causal_mask = causal_masks[:,
                                          :math.prod(Fused_st_extent),
                                          :math.prod(Fused_st_extent)]
                
                for i in range(self.st_mha_blocks):
            
                    Fused_st, aux_loss, _ , _ = getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st,
                                                                   causal_mask,
                                                                   mc_dropout)
                    total_aux_loss.append(aux_loss)
                
                
                Fused_st = torch.reshape(Fused_st, (Fused_st.shape[0],*Fused_st_extent,Fused_st.shape[-1]))        
                
                Fused_st = self.linear(Fused_st)
                Fused_st = Fused_st + Autoreg_st
            
                Output_st = self.output(Fused_st)
            
                Sparse_data = torch.cat([Z.clone(),
                                       Output_st[:,-1,:,:]],
                                      dim=-1)

                Sparse_data_mask = torch.ones((Sparse_data[:,:,0].shape)).to(torch.bool).to(Sparse_data.device)
            
            
            Autoreg_aux_loss = torch.tensor(Autoreg_aux_loss).mean()
            Weather_aux_loss = torch.tensor(Weather_aux_loss).mean()
            total_aux_loss = torch.tensor(total_aux_loss).mean()
            total_aux_loss += Autoreg_aux_loss + Weather_aux_loss
            
            Output_st = Output_st.squeeze()
            
            if get_aux_loss is True:
                    return [Output_st, total_aux_loss]
            else:
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
                 STMoE_prenorm = False):
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
        
        
        moe = MoE_alt(
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
        
        output = nn.functional.dropout1d(output.permute((0,2,1)),
                                        p = self.dropout_p, training = mc_dropout)    
        output = output.permute((0,2,1))
        
        return output, total_aux_loss, balance_loss, router_z_loss

class SparseData_STMoE(nn.Module):
    
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
                                                            in_channels = 6,
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
        
        self.Fusion_Embedding = nn.Sequential(nn.Linear(self.spatial_embedding_dim,
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
                                   
        self.output = nn.Sequential(nn.Linear(self.spatial_embedding_dim,
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
                
                Weather_Keys = W[0][:,:3,timestep,:,:].flatten(start_dim = -2, end_dim = -1).permute((0,2,1)) # 
                Weather_Values = W[0][:,:,timestep,:,:].flatten(start_dim = -2, end_dim = -1).permute((0,2,1))
                Weather_st.append(self.Weather_Module(
                                                    K = Weather_Keys,
                                                    V = Weather_Values,
                                                    Q = Z
                                                    ))
                
            Autoreg_st = torch.stack(Autoreg_st, dim = 1)
            Weather_st = torch.stack(Weather_st, dim = 1)

            # Fused_st = torch.cat([Autoreg_st,
            #                         Weather_st], 
            #                         dim = -1)
            
            Fused_st = Autoreg_st + Weather_st
            
            Fused_st_extent = Fused_st.shape[1:3]
            Fused_st = torch.flatten(Fused_st, 1, 2)
            
            Fused_st =  nn.functional.dropout1d(Fused_st.permute((0,2,1)),
                                                p = self.spatial_dropout, training = True)
            
            Fused_st = self.Fusion_Embedding(Fused_st.permute((0,2,1)))
            
            Fused_st += ST_Conditionings
            
            total_aux_loss = []
            
            for i in range(self.st_mha_blocks):
            
                Fused_st, aux_loss, _, _= getattr(self, f"MHA_STMoE_Block_{i}")(Fused_st, causal_masks,
                                                                                mc_dropout = True)
                total_aux_loss.append(aux_loss)
            
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
                
                Autoreg_st = torch.stack(Autoreg_st_rlist, dim = 1)
                Weather_st = torch.stack(Weather_st_rlist, dim = 1)
                # Fused_st = torch.cat([Autoreg_st,
                #                     Weather_st], 
                #                     dim = -1)
                
                Fused_st = Autoreg_st + Weather_st
                
                Fused_st_extent = Fused_st.shape[1:3]
                Fused_st = torch.flatten(Fused_st, 1, 2)
                
                Fused_st =  nn.functional.dropout1d(Fused_st.permute((0,2,1)),
                                                p = self.spatial_dropout, training = mc_dropout)
                
                Fused_st = self.Fusion_Embedding(Fused_st.permute((0,2,1)))
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
                                STMoE_prenorm = True)
        
        self.ST_Conditioning_Module = ST_Conditioning_Block(
                                                            in_channels = 5,
                                                            hidden_channels = 32, 
                                                            out_channels = 1, 
                                                            activation = self.activation,
                                                            LayerNorm = True) 

        ### Weather module ### 
        
        self.Weather_Module = Spatial_Attention_Block_MoE(
                 embedding_dim = self.spatial_embedding_dim,
                 input_channles = self.weather_dim[0],
                 heads = self.spatial_heads,
                 num_experts= self.num_experts,
                 output_channels = self.spatial_embedding_dim,
                 activation=self.activation,
                 elementwise_affine=self.layernorm_affine,
                 STMoE_prenorm = True)
        
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
                                STMoE_prenorm = True)
        
                                   
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
    