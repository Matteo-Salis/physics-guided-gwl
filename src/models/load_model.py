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
                upsampling_dim = [104, 150]), model_name
        
    elif config["model"] == "VideoCB_ConvLSTM":
        
        model_name = "VideoCB_ConvLSTM"
        print(f"Model: {model_name}")
        
        return VideoCB_ConvLSTM(
                weather_CHW_dim = config["weather_CHW_dim"],
                cb_emb_dim = config["cb_emb_dim"],
                cb_heads = config["cb_heads"],
                channels_cb_wb = config["channels_cb_wb"],
                convlstm_input_units = config["convlstm_input_units"],
                convlstm_units = config["convlstm_units"],
                convlstm_kernel = config["convlstm_kernel"],
                densification_dropout = config["densification_dropout"],
                upsampling_dim = [104, 150]), model_name
    
    else:
        raise Exception("Model name unknown.")
    
    
    
if __name__ == "__main__":
    pass