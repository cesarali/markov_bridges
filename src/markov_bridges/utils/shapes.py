import torch

right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
right_time_size = lambda t,x: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

def where_to_go_x(x,vocab_size):
    x_to_go = torch.arange(0, vocab_size)
    x_to_go = x_to_go[None, None, :].repeat((x.size(0), x.size(1), 1)).float()
    x_to_go = x_to_go.to(x.device)
    return x_to_go

def nodes_and_edges_masks(nodesxsample,max_n_nodes,device=None):
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1
    # Compute edge_mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)
    node_mask = node_mask.unsqueeze(2)

    if device is not None:
        node_mask = node_mask.to(device)
        edge_mask = edge_mask.to(device)
    return node_mask,edge_mask