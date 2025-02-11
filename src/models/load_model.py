from models.models_1d import *
from models.models_2d import *

def load_model(config):
    timesteps = config["timesteps"]

    if config["model"] == "Discrete2DConcat1":
        model_name = "Concat1_2D"
        return Discrete2DConcat1(timesteps), model_name
    
    elif config["model"] == "Discrete2DConcat16":
        model_name = "Concat16_2D"
        return Discrete2DConcat16(timesteps), model_name
    
    elif config["model"] == "Discrete2DConcat1_Time":
        model_name = "Concat1_T_2D"
        return Discrete2DConcat1_Time(timesteps), model_name
    
    elif config["model"] == "Discrete2DConvLSTM":
        model_name = "ConvLSTM_2D"
        return Discrete2DConvLSTM(timesteps), model_name
    
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
                 ph_params = ph_params), model_name
    
    elif dict_files["model"] == "CCNN_att":
        
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
        
    elif dict_files["model"] == "SC_LSTM_att":
        
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
    
    else:
        raise Exception("Model name unknown.")