import torch
from torch.optim import lr_scheduler

def load_optimizer(config, model):
    
    if config["optimizer"] == "adam" or config["optimizer"] is None:
        print(f"Optimizer: Adam selected - lr:{config['lr']}")
        return torch.optim.Adam(model.parameters(),
                                 lr=config['lr'])
        
    elif config["optimizer"] == "adamw":
        print(f"Optimizer: AdamW selected - lr:{config['lr']} - weight_decay: {config['weight_decay']}")
        return torch.optim.AdamW(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay = config['weight_decay'])
        
    elif config["optimizer"] == "sgd":
        
        print("Optimizer: SGD", end = " ")
        
        if config["momentum"]:
            print(f"with {config['momentum']} Momentum", end = " ")
        else:
            config["momentum"] = 0
            
        if config["nesterov"]:
            print("with Nesterov", end = " ")
        else:
            config["nesterov"] = False
            
            
        print(f"selected - lr:{config['lr']} - weight_decay:{config['weight_decay']}")
        
        return torch.optim.SGD(model.parameters(),
                               lr=config["lr"],
                               momentum=config["momentum"],
                               weight_decay=config["weight_decay"],
                               nesterov=config["nesterov"])
        
        
        
def load_lr_scheduler(config, optimizer):
    if config["lr_scheduling"] == "linear":
        pass
    elif config["lr_scheduling"] == "exponential":
        pass
    elif isinstance(config["lr_scheduling"], list):
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_scheduling"][1], gamma=config["lr_scheduling"][0])