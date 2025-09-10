from models.models_ST_MultiPoint import *

def load_model(config):

    
    if config["model"] == "ST_MultiPoint_Net":
        
        model_name = "ST_MultiPoint_Net"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_Net(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                joint_mod_blocks = config["joint_mod_blocks"],
                joint_mod_heads = config["joint_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"],
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"])
        
    elif config["model"] == "ST_MultiPoint_Net_SAGW":
        
        model_name = "ST_MultiPoint_Net_SAGW"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_Net_SAGW(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                joint_mod_blocks = config["joint_mod_blocks"],
                joint_mod_heads = config["joint_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"],
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"])
        
    elif config["model"] == "ST_MultiPoint_STNet_SAGW":
        
        model_name = "ST_MultiPoint_STNet_SAGW"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_STNet_SAGW(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                joint_mod_blocks = config["joint_mod_blocks"],
                joint_mod_heads = config["joint_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                densification_dropout_p = config["densification_dropout_p"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"],
                simplified_embedding= config["simplified_embedding"])
        
    elif config["model"] == "ST_MultiPoint_STDisNet_SAGW_K":
        
        model_name = "ST_MultiPoint_STDisNet_SAGW_K"
        print(f"Model: {model_name}")
    
        model = ST_MultiPoint_STDisNet_SAGW_K(value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                s_coords_dim = config["s_coords_input_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                densification_dropout_p = config["densification_dropout_p"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"],
                simplified_embedding= config["simplified_embedding"])
        
    elif config["model"] == "ST_MultiPoint_Net_OnlyLag":
    
        model_name = "ST_MultiPoint_Net_OnlyLag"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_Net_OnlyLag(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                joint_mod_blocks = config["joint_mod_blocks"],
                joint_mod_heads = config["joint_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                    config["weather_lags"]+1],
                dropout = config["dropout"],
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"])
        
    
    elif config["model"] == "ST_MultiPoint_DisNet":
        
        model_name = "ST_MultiPoint_DisNet"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_DisNet(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"], 
                activation = config["activation"])
        
    elif config["model"] == "ST_MultiPoint_DisNet_K":
        
        model_name = "ST_MultiPoint_DisNet_K"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_DisNet_K(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                s_coords_dim = config["s_coords_input_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"], 
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"])
        
    elif config["model"] == "ST_MultiPoint_DisNet_SAGW_K":
        
        model_name = "ST_MultiPoint_DisNet_SAGW_K"
        print(f"Model: {model_name}")
    
        model = ST_MultiPoint_DisNet_SAGW_K(value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                s_coords_dim = config["s_coords_input_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"], 
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"])
        
    elif config["model"] == "ST_MultiPoint_DisNet_SAGW_K_ALT":
        
        model_name = "ST_MultiPoint_DisNet_SAGW_K_ALT"
        print(f"Model: {model_name}")
    
        model = ST_MultiPoint_DisNet_SAGW_K_ALT(value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                s_coords_dim = config["s_coords_input_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"], 
                densification_dropout_p = config["densification_dropout_p"],
                densification_dropout_dv = config["fill_value"],
                activation = config["activation"],
                emb_W=config["emb_W"],
                normalization = config["model_normalization"])
        
    elif config["model"] == "ST_MultiPoint_DisNet_alt":
        
        model_name = "ST_MultiPoint_DisNet_alt"
        print(f"Model: {model_name}")
        
        model = ST_MultiPoint_DisNet_alt(
                value_dim_GW = config["GW_value_input_dim"],
                value_dim_Weather = config["Weather_value_input_dim"], 
                embedding_dim = config["embedding_dim"],
                st_coords_dim = config["st_coords_input_dim"],
                spatial_mha_heads = config["spatial_mha_heads"],
                displacement_mod_blocks = config["displacement_mod_blocks"],
                displacement_mod_heads = config["displacement_mod_heads"],
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                dropout = config["dropout"], 
                activation = config["activation"])
    
    elif config["model"] == "ST_MultiPoint_PhyDisNet":
        pass
    
    else:
        raise Exception("Model name unknown.")
    
    
    
    if config["pretrain_model"] is not None:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(config["pretrain_model"],
                                 weights_only=True), strict=False)
        print("Done!")
    
    
    if config["model_init"] == "He_uniform":
        print("He Initialization Applied.")
        #model = model.apply(partial(weight_init_he, activation = config["activation"], distribution = "uniform"))
        weight_init_He_alt(model,
                          config["activation"])
    
    elif config["model_init"] == "He_normal":
        print("He Initialization Applied.")
        model = model.apply(partial(weight_init_he, activation = config["activation"], distribution = "normal"))
    
    elif config["model_init"] == "Ortho":
        print("Orthogonal Initialization Applied.")
        #model = model.apply(partial(weight_init_ortho, activation = config["activation"]))
        weight_init_ortho_alt(model,
                          config["activation"])
    return model, model_name
    
    
if __name__ == "__main__":
    pass
