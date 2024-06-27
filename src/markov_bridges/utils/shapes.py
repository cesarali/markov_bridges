import torch

right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = lambda x,t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

def where_to_go_x(x,vocab_size):
    x_to_go = torch.arange(0, vocab_size)
    x_to_go = x_to_go[None, None, :].repeat((x.size(0), x.size(1), 1)).float()
    x_to_go = x_to_go.to(x.device)
    return x_to_go