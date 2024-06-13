import torch

def sample_from_dataloader_iterator(dataloder_iterator,sample_size,flatten=True):
    """
    Samples data from the dataloader until the sample_size is met.
    
    Args:
        dataloader (DataLoader): The dataloader to sample from.
        sample_size (int): The total number of samples to collect.
        flatten (bool): Whether to flatten the samples or not.
    
    Returns:
        torch.Tensor: A tensor containing the collected samples.
    """
    size_left = sample_size
    x_0 = []
    while size_left > 0:
        dataloader_iterator = iter(dataloder_iterator)
        for databatch in dataloader_iterator:
            x_ = databatch[0]
            batch_size = x_.size(0)
            take_size = min(size_left, batch_size)
            x_0.append(x_[:take_size])
            size_left -= take_size
            if size_left == 0:
                break

    x_0 = torch.vstack(x_0)
    actual_sample_size = x_0.size(0)
    if flatten:
        x_0 = x_0.reshape(actual_sample_size, -1)
    return x_0