from utils.training_1d import *

def test_model(config):

    if config["loss"] == "data":
        return test_dl_model_1d
    
    