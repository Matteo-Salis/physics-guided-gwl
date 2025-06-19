from models.models_1d import *
from models.models_2d import *
from models.models_2D import *

def load_model(config):
    twindow = config["twindow"]

    if config["model"] == "Discrete2DConcat1":
        model_name = "Concat1_2D"
        return Discrete2DConcat1(twindow), model_name
    
    elif config["model"] == "Discrete2DConcat16":
        model_name = "Concat16_2D"
        return Discrete2DConcat16(twindow), model_name
    
    elif config["model"] == "Discrete2DConcat1_Time":
        model_name = "Concat1_T_2D"
        return Discrete2DConcat1_Time(twindow), model_name
    
    elif config["model"] == "Discrete2DConvLSTM":
        model_name = "ConvLSTM_2D"
        return Discrete2DConvLSTM(twindow), model_name
    
    elif config["model"] == "PICCNN_att_1D":
        
        print("model physics informed causal cnn att")
        ph_params = {"hyd_cond": [config["pde_hyd_cond"][0],
                              config["pde_hyd_cond"][1],
                              config["pde_hyd_cond"][2]]}
        
        model_name = "PICCNN_A_1D"
        
        return SC_PICCNN_att(timestep = config["timesteps"],
                 cb_emb_dim = config["cb_emb_dim"],
                 cb_att_h = config["cb_att_h"],
                 cb_fc_layer = config["cb_fc_layer"],
                 cb_fc_neurons = config["cb_fc_neurons"],
                 conv_filters = config["conv_filters"],
                 ccnn_input_filters =  config["ccnn_input_filters"],
                 ccnn_kernel_size =  config["ccnn_kernel_size"],
                 ccnn_n_filters =  config["ccnn_n_filters"],
                 ccnn_n_layers =  config["ccnn_n_layers"],
                 ph_params = ph_params,
                 ph_params_neurons = config["ph_params_neurons"]), model_name
    
    elif config["model"] == "CCNN_att_1D":
        
        print("model causal cnn att")
        
        model_name = "CCNN_A_1D"
        
        return SC_CCNN_att(timestep = config["timesteps"],
                    cb_emb_dim = config["cb_emb_dim"],
                    cb_att_h = config["cb_att_h"],
                    cb_fc_layer = config["cb_fc_layer"],
                    cb_fc_neurons = config["cb_fc_neurons"],
                    conv_filters = config["conv_filters"],
                    ccnn_input_filters =  config["ccnn_input_filters"],
                    ccnn_kernel_size =  config["ccnn_kernel_size"],
                    ccnn_n_filters =  config["ccnn_n_filters"],
                    ccnn_n_layers =  config["ccnn_n_layers"],
                    ), model_name
        
    elif config["model"] == "CCNN_idw_1D":
        
        print("model causal cnn idw")
        
        model_name = "CCNN_IDW_1D"
        
        return SC_CCNN_idw(timestep = config["timesteps"],
                    cb_fc_layer = config["cb_fc_layer"],
                    cb_fc_neurons = config["cb_fc_neurons"],
                    conv_filters = config["conv_filters"],
                    ccnn_input_filters =  config["ccnn_input_filters"],
                    ccnn_kernel_size =  config["ccnn_kernel_size"],
                    ccnn_n_filters =  config["ccnn_n_filters"],
                    ccnn_n_layers =  config["ccnn_n_layers"],
                    ), model_name
        
    elif config["model"] == "CCNN_att_TRPS_1D":
    
        print("model causal cnn att blocks")
        
        model_name = "CCNN_A_1D"
    
        return SC_CCNN_att_TRSP(timestep = config["timesteps"],
                    cb_emb_dim = config["cb_emb_dim"],
                    cb_att_h = config["cb_att_h"],
                    cb_fc_layer = config["cb_fc_layer"],
                    cb_fc_neurons = config["cb_fc_neurons"],
                    conv_filters = config["conv_filters"],
                    ccnn_input_filters =  config["ccnn_input_filters"],
                    ccnn_kernel_size =  config["ccnn_kernel_size"],
                    ccnn_n_filters =  config["ccnn_n_filters"],
                    ccnn_n_layers =  config["ccnn_n_layers"],
                                ), model_name
    
    elif config["model"] == "LSTM_att_1D":
        
        print("model lstm att")
        
        model_name = "LSTM_A_1D"
        
        return SC_LSTM_att(timestep = config["timesteps"],
                    cb_emb_dim = config["cb_emb_dim"],
                    cb_att_h = config["cb_att_h"],
                    cb_fc_layer = config["cb_fc_layer"],
                    cb_fc_neurons = config["cb_fc_neurons"],
                    conv_filters = config["conv_filters"],
                    lstm_layer = config["lstm_layer"],
                    lstm_input_units = config["lstm_input_units"],
                    lstm_units = config["lstm_units"]
                    ), model_name
        
    elif config["model"] == "LSTM_att_TT_1D":
        
        print("model lstm att Teacher Training")
        
        model_name = "LSTM_A_TT_1D"
        
        return SC_LSTM_att_TT(timestep = config["timesteps"],
                    cb_emb_dim = config["cb_emb_dim"],
                    cb_att_h = config["cb_att_h"],
                    cb_fc_layer = config["cb_fc_layer"],
                    cb_fc_neurons = config["cb_fc_neurons"],
                    conv_filters = config["conv_filters"],
                    lstm_layer = config["lstm_layer"],
                    lstm_input_units = config["lstm_input_units"],
                    lstm_units = config["lstm_units"]
                    ), model_name
        
    elif config["model"] == "LSTM_idw_1D":
        
        print("model lstm idw")
        
        model_name = "LSTM_IDW_1D"
        
        return SC_LSTM_idw(timestep = config["timesteps"],
                           cb_fc_layer = config["cb_fc_layer"],
                           cb_fc_neurons = config["cb_fc_neurons"],
                           conv_filters = config["conv_filters"],
                           lstm_layer = config["lstm_layer"],
                           lstm_input_units = config["lstm_input_units"],
                           lstm_units = config["lstm_units"]
                           ), model_name
        
    elif config["model"] == "AttCB_ConvLSTM":
        
        model_name = "AttCB_ConvLSTM"
        print(f"Model: {model_name}")
        
        return AttCB_ConvLSTM(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb_wb = config["channels_cb_wb"],
                convlstm_input_units = config["convlstm_input_units"],
                convlstm_units = config["convlstm_units"],
                convlstm_kernel = config["convlstm_kernel"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"]), model_name
        
    elif config["model"] == "VideoCB_ConvLSTM":
        
        model_name = "VideoCB_ConvLSTM"
        print(f"Model: {model_name}")
        
        return VideoCB_ConvLSTM(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb = config["channels_cb"],
                channels_wb = config["channels_wb"],
                convlstm_IO_units = config["convlstm_IO_units"],
                convlstm_hidden_units = config["convlstm_hidden_units"],
                convlstm_kernel = config["convlstm_kernel"],
                convlstm_nlayer = config["convlstm_nlayer"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"]), model_name
        
    elif config["model"] == "FullAttention_ConvLSTM":
        
        model_name = "FullAttention_ConvLSTM"
        print(f"Model: {model_name}")
        
        return FullAttention_ConvLSTM(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb = config["channels_cb"],
                channels_wb = config["channels_wb"],
                convlstm_IO_units = config["convlstm_IO_units"],
                convlstm_hidden_units = config["convlstm_hidden_units"],
                convlstm_kernel = config["convlstm_kernel"],
                convlstm_nlayer = config["convlstm_nlayer"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"]), model_name
        
    elif config["model"] == "VideoCB_ConvLSTM_PI":
        
        model_name = "VideoCB_ConvLSTM"
        print(f"Model: {model_name}")
        
        return VideoCB_ConvLSTM_PI(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb = config["channels_cb"],
                channels_wb = config["channels_wb"],
                convlstm_IO_units = config["convlstm_IO_units"],
                convlstm_hidden_units = config["convlstm_hidden_units"],
                convlstm_kernel = config["convlstm_kernel"],
                convlstm_nlayer = config["convlstm_nlayer"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"]), model_name
        
    elif config["model"] == "FullAttention_CausalConv":
        
        model_name = "FullAttention_CausalConv"
        print(f"Model: {model_name}")
        
        return FullAttention_CausalConv(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb = config["channels_cb"],
                channels_wb = config["channels_wb"],
                cconv3d_input_channels = config["cconv3d_input_channels"],
                cconv3d_hidden_channels = config["cconv3d_hidden_channels"],
                cconv3d_kernel = config["cconv3d_kernel"],
                cconv3d_dilation = config["cconv3d_dilation"],
                cconv3d_layers = config["cconv3d_layers"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"],
                activation = config["activation"]), model_name
        
        
    elif config["model"] == "FullAttention_ViViT":
        
        model_name = "FullAttention_ViViT"
        print(f"Model: {model_name}")
        
        return FullAttention_ViViT(
                weather_CHW_dim = config["weather_CHW_dim"],
                sparse_emb_dim = config["sparse_emb_dim"],
                sparse_heads = config["sparse_heads"],
                dense_emb_dim = config["dense_emb_dim"],
                dense_heads = config["dense_heads"],
                patch_size = config["patch_size"],
                mha_blocks = config["mha_blocks"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = config["upsampling_dim"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"]), model_name
        
        
    elif config["model"] == "SparseData_Transformer":
        
        model_name = "SparseData_Transformer"
        print(f"Model: {model_name}")
        
        return SparseData_Transformer(
                weather_CHW_dim = config["weather_CHW_dim"],
                target_dim = config["target_dim"],
                spatial_embedding_dim = config["spatial_embedding_dim"],
                spatial_heads = config["spatial_heads"],
                fusion_embedding_dim = config["fusion_embedding_dim"],
                st_heads = config["st_heads"],
                st_mha_blocks = config["st_mha_blocks"],
                densification_dropout = config["densification_dropout"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"],
                activation= config["activation"]), model_name
        
    elif config["model"] == "SparseData_STMoE":
        
        model_name = "SparseData_STMoE"
        print(f"Model: {model_name}")
        
        return SparseData_STMoE(
                weather_CHW_dim = config["weather_CHW_dim"],
                target_dim = config["target_dim"],
                spatial_embedding_dim = config["spatial_embedding_dim"],
                spatial_heads = config["spatial_heads"],
                fusion_embedding_dim = config["fusion_embedding_dim"],
                st_heads = config["st_heads"],
                st_mha_blocks = config["st_mha_blocks"],
                num_experts = config["num_experts"],
                densification_dropout = config["densification_dropout"],
                layernorm_affine = config["layernorm_affine"],
                spatial_dropout = config["spatial_dropout"],
                activation= config["activation"]), model_name
    
    else:
        raise Exception("Model name unknown.")
    
    
    
if __name__ == "__main__":
    pass