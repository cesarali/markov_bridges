import torch
from torchdyn.core import NeuralODE

from markov_bridges.configs.config_classes.generative_models.cfm_config import CFMConfig
from markov_bridges.models.generative_models.cfm_forward import ContinuousForwardMap
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple


def ODESamplerCFM(config: CFMConfig,
                  drift_model: ContinuousForwardMap,
                  x_0: MarkovBridgeDataNameTuple):
    """
    :param drift_model:
    :param x_0:
    :param N:
    :return:
    """

    num_steps = config.pipeline.number_of_steps
    ode_solver = config.pipeline.ode_solver
    atol = config.pipeline.atol
    rtol = config.pipeline.rtol
    sensitivity = config.pipeline.sensitivity
    device = x_0.source_continuous.device
    time_steps = torch.linspace(0, 1, num_steps, device=device)

    drift = ContextWrapper(drift_model.continuous_network, 
                           context_discrete=x_0.context_discrete if config.data.has_context_discrete else None, 
                           context_continuous=x_0.context_continuous if config.data.has_context_continuous else None)

    if ode_solver == 'euler':
        node = EulerODESolver(vector_field=drift)

    elif ode_solver == 'rk4':
        node = RungeKuttaODESolver(vector_field=drift)
    
    elif ode_solver == 'midpoint':
        node = MidpointODESolver(vector_field=drift)
    
    elif 'torchdyn' in ode_solver:
        ode_solver = ode_solver.split('_')[-1]
        node = NeuralODE(vector_field=drift, 
                         solver=ode_solver, 
                         sensitivity=sensitivity, 
                         seminorm=True if ode_solver=='dopri5' else False,
                         atol=atol if ode_solver=='dopri5' else None,
                         rtol=rtol if ode_solver=='dopri5' else None)
    
    else:
        raise ValueError(f'ODE solver {ode_solver} not supported')
    
    trajectories = node.trajectory(x=x_0.source_continuous, t_span=time_steps).detach().cpu()
    trajectories = trajectories.detach().float()

    return trajectories


class ContextWrapper(torch.nn.Module):
    """ Wraps time-dependent model to include context 
    """
    def __init__(self, net, context_discrete=None, context_continuous=None):
        super().__init__()
        self.nn = net
        self.context_discrete = context_discrete
        self.context_continuous = context_continuous

    def reshape_time_like(self, t, x):
        if isinstance(t, (float, int)): return t
        else: return t.reshape(-1, *([1] * (x.dim() - 1)))

    def forward(self, t, x, args):
        t = t.repeat(x.shape[0])
        t = self.reshape_time_like(t, x)
        return self.nn(x_continuous=x, 
                       times=t, 
                       context_discrete=self.context_discrete, 
                       context_continuous=self.context_continuous)

# Native ODE solver methods:

class EulerODESolver:
    def __init__(self, vector_field):
        self.vector_field = vector_field

    def trajectory(self, x, t_span, *args):
        time_steps = len(t_span)
        dt = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i-1]
            x = x + dt * self.vector_field(t, x, args).to(x.device)
            trajectory.append(x)

        return torch.stack(trajectory)
    

class MidpointODESolver:
    def __init__(self, vector_field):
        self.vector_field = vector_field

    def trajectory(self, x, t_span, *args):
        time_steps = len(t_span)
        dt = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i - 1]
            k1 = self.vector_field(t, x, args)
            x_mid = x + 0.5 * dt * k1
            k2 = self.vector_field(t + 0.5 * dt, x_mid, args)
            x_next = x + dt * k2
            trajectory.append(x_next)
            x = x_next

        return torch.stack(trajectory)
    
class RungeKuttaODESolver:
    def __init__(self, vector_field):
        self.vector_field = vector_field

    def trajectory(self, x, t_span, *args):
        time_steps = len(t_span)
        dt = (t_span[-1] - t_span[0]) / (time_steps - 1)
        trajectory = [x]

        for i in range(1, time_steps):
            t = t_span[i - 1]
            k1 = dt * self.vector_field(t, x, args)
            k2 = dt * self.vector_field(t + 0.5 * dt, x + 0.5 * k1, args)
            k3 = dt * self.vector_field(t + 0.5 * dt, x + 0.5 * k2, args)
            k4 = dt * self.vector_field(t + dt, x + k3, args)

            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trajectory.append(x)

        return torch.stack(trajectory)

