import torch

def nametuple_to_device(obj, device):
    for attribute in vars(obj):
        value = getattr(obj, attribute)
        if isinstance(value, torch.Tensor):
            setattr(obj, attribute, value.to(device))