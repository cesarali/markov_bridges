from torch.nn import functional as F

def categorical_counts_per_path(x_path,vocab_size=3,normalize=True):
    """
    parameters
    ----------
    x_path: torch.Tensor

    return
    ------
    sample_size,paths_counts
    """
    sample_size = x_path.shape[0]
    paths_counts = F.one_hot(x_path.long(),num_classes=vocab_size).sum(axis=0)
    if normalize:
        paths_counts = paths_counts/sample_size
    return paths_counts