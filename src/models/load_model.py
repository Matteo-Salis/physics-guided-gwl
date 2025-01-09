from models.load_models_1d import *
from models.load_models_2d import *

def load_model(config):
    timesteps = config["timesteps"]

    if config["model"] == "Discrete2DMidConcatNN":
        return Discrete2DMidConcatNN(timesteps)
    elif config["model"] == "Discrete2DNN":
        return Discrete2DNN(timesteps)
    else:
        raise Exception("Model name unknown.")