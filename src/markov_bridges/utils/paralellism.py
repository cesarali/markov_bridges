import torch

def nametuple_to_device(databatch, device='cpu'):
    new_databatch = databatch._asdict()
    for key, value in new_databatch.items():
        new_databatch[key] = value.to(device)
    databatch_nametuple = type(databatch)
    modified_databatch = databatch_nametuple(**new_databatch)
    return modified_databatch


def check_model_devices(x):
    return x.parameters().__next__().device
