
import torch

#******************************************************************************* 

def load_model(model, model_path, target_device):
    model.load_state_dict(torch.load(model_path, map_location=target_device))
    return model

#******************************************************************************* 