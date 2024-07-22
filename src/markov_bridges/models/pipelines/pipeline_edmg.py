
from markov_bridges.data.qm9.sampling import sample_chain

def save_and_sample_chain(model, args, device, dataset_info, prop_dist):
    one_hot, charges, x = sample_chain(config=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    return one_hot, charges, x