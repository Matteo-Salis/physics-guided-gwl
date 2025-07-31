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
                displacement_mod_blocks = 1,
                displacement_mod_heads = 2,
                GW_W_temp_dim = [len(config["target_lags"]),
                                 config["weather_lags"]+1],
                # densification_dropout = ,
                dropout = config["dropout"], 
                activation = config["activation"])
    
    else:
        raise Exception("Model name unknown.")
    
    
    
    if config["pretrain_model"] is not None:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(config["pretrain_model"],
                                 weights_only=True), strict=False)
        print("Done!")
    
    
    return model, model_name
    
    
if __name__ == "__main__":
    pass