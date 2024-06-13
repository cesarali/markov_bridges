
def check_model_devices(x):
    return x.parameters().__next__().device