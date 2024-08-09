import torch

def nametuple_to_device(obj, device):
    for attribute in vars(obj):
        value = getattr(obj, attribute)
        if isinstance(value, torch.Tensor):
            setattr(obj, attribute, value.to(device))

#======================================================================
# VARIANCE
#======================================================================
def compute_first_moment(self,t,x1,x0):
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x0.size(0),), t).to(x0.device)

    t = right_time_size(t).to(x0.device)
    t1 = right_time_size(1.).to(x0.device)
    t0 = right_time_size(0.).to(x0.device)

    i = x1
    j = x0

    S = self.vocab_size
    integral_t0 = self.beta_integral(t, t0)
    integral_1t = self.beta_integral(t1, t)
    integral_10 = self.beta_integral(t1, t0)

    w_t0 = torch.exp(-S * integral_t0)[:,None,None]
    w_1t = torch.exp(-S * integral_1t)[:,None,None]
    w_10 = torch.exp(-S * integral_10)[:,None,None]
    # Kronecker delta in PyTorch
    kronecker_delta_ij = (i == j).float()[:,:,None]
    i = i[:,:,None]
    j = j[:,:,None]

    # Precompute common terms to simplify the expression
    term_S1 = S + 1  # Term involving S
    part1 = w_10 * (S * kronecker_delta_ij - 1) + 1
    part2 = S * w_10 * kronecker_delta_ij - w_10 + 1

    # Calculate each term of the first moment expression
    term1 = part1 * (S + w_1t * w_t0 * term_S1 - w_1t * term_S1 - w_t0 * term_S1 + 1) / 2
    term2 = part2 * (
                S * i * w_1t * w_t0 * kronecker_delta_ij - i * w_1t * w_t0 + i * w_1t - j * w_1t * w_t0 + j * w_t0)

    # Combine terms to compute the first moment
    first_moment = (term1 + term2) / (part1 * part2)
    return first_moment

def compute_second_moment(self,t,x1,x0):
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x0.size(0),), t).to(x0.device)

    t = right_time_size(t).to(x0.device)
    t1 = right_time_size(1.).to(x0.device)
    t0 = right_time_size(0.).to(x0.device)

    i = x1
    j = x0

    S = self.vocab_size
    integral_t0 = self.beta_integral(t, t0)
    integral_1t = self.beta_integral(t1, t)
    integral_10 = self.beta_integral(t1, t0)

    w_t0 = torch.exp(-S * integral_t0)[:,None,None]
    w_1t = torch.exp(-S * integral_1t)[:,None,None]
    w_10 = torch.exp(-S * integral_10)[:,None,None]
    # Kronecker delta in PyTorch
    kronecker_delta_ij = (i == j).float()[:,:,None]
    i = i[:,:,None]
    j = j[:,:,None]

    # Precompute common terms to simplify the expression
    term_S0 = 2 * S ** 2 + 3 * S + 1  # Term involving S
    part1 = w_10 * (S * kronecker_delta_ij - 1) + 1
    part2 = S * w_10 * kronecker_delta_ij - w_10 + 1

    # Calculate each term of the second moment expression
    term1 = part1 * (term_S0 + w_1t * w_t0 * term_S0 - w_1t * term_S0 - w_t0 * term_S0 + 1) / 6
    term2 = part2 * (
                S * i ** 2 * w_1t * w_t0 * kronecker_delta_ij - i ** 2 * w_1t * w_t0 + i ** 2 * w_1t - j ** 2 * w_1t * w_t0 + j ** 2 * w_t0)

    # Combine terms to compute the second moment
    second_moment = (term1 + term2) / (part1 * part2)
    return second_moment

def compute_variance_torch(self,t,x1,x0):

    mean = self.compute_first_moment(t,x1,x0)
    second_moment = self.compute_second_moment(t,x1,x0)

    return second_moment - mean**2

