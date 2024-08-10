
import torch
from markov_bridges.models.deprecated.generative_models.cmb import CMB
from markov_bridges.configs.config_classes.generative_models.cmb_config import CMBConfig
from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.models.networks.temporal.mixed.mixed_networks_utils import load_mixed_network


def test_cmb_loss():
    model_config = CMBConfig(continuous_loss_type="drift")
    model_config.data = IndependentMixConfig(has_context_discrete=True)

    cmb = CMB(model_config,device=torch.device("cpu"))
    databatch = cmb.dataloader.get_databatch()

    print(databatch.time.shape)
    discrete_sample,continuous_sample = cmb.forward_map.sample_bridge(databatch)
    loss = cmb.forward_map.loss(databatch,discrete_sample,continuous_sample)
    print(loss)

if __name__=="__main__":
    test_cmb_loss()

