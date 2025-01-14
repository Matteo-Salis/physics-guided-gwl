from models.load_models_1d import *
from models.load_models_2d import *

def load_model(config):
    timesteps = config["timesteps"]

    if config["model"] == "Discrete2DConcat1Sum":
        return Discrete2DConcat1Sum(timesteps)
    elif config["model"] == "Discrete2DSumSuperRes":
        return Discrete2DSumSuperRes(timesteps)
    elif config["model"] == "Discrete2DConcat1":
        return Discrete2DConcat1(timesteps)
    elif config["model"] == "Discrete2DConcat16":
        return Discrete2DConcat16(timesteps)
    else:
        raise Exception("Model name unknown.")