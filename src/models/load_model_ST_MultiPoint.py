from models.models_ST_MultiPoint import *

def load_model(config):

        
    if config["model"] == "STAINet":
        
        model_name = "STAINet"
        print(f"Model: {model_name}")
        
        model = STAINet(
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
        
    elif config["model"] == "PSTAINet_IB":
        
        model_name = "PSTAINet_IB"
        print(f"Model: {model_name}")
    
        model = PSTAINet_IB(value_dim_GW = config["GW_value_input_dim"],
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
