import torch
from markov_bridges.models.pipelines.thermostats import Thermostat


def conditional_transition_rate(self, x, x1, t, thermostat:Thermostat):
    """
    \begin{equation}
    f_t(\*x'|\*x,\*x_1) = \frac{p(\*x_1|x_t=\*x')}{p(\*x_1|x_t=\*x)}f_t(\*x'|\*x)
    \end{equation}
    """
    x_to_go = self.where_to_go_x(x)

    P_xp_to_x1 = self.conditional_probability(x1, x_to_go, t=1., t0=t)
    P_x_to_x1 = self.conditional_probability(x1, x, t=1., t0=t)

    forward_rate = thermostat(t)[:,None,None]
    rate_transition = (P_xp_to_x1 / P_x_to_x1) * forward_rate

    return rate_transition

def multivariate_telegram_conditional(x, x0, t, t0,vocab_size,thermostat:Thermostat):
    """

    \begin{equation}
    P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
    \end{equation}

    \begin{equation}
    w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
    \end{equation}

    """
    right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

    t = right_time_size(t).to(x0.device)
    t0 = right_time_size(t0).to(x0.device)

    integral_t0 = thermostat.beta_integral(t, t0)
    w_t0 = torch.exp(-vocab_size * integral_t0)

    x = right_shape(x)
    x0 = right_shape(x0)

    delta_x = (x == x0).float()
    probability = 1. / vocab_size + w_t0[:, None, None] * ((-1. / vocab_size) + delta_x)
    return probability
