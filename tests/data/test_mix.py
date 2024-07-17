from markov_bridges.configs.config_classes.data.basics_configs import IndependentMixConfig
from markov_bridges.data.categorical_samples import IndependentMixDataloader
from dataclasses import asdict
from pprint import pprint

if __name__=="__main__":
    data_config = IndependentMixConfig()
    pprint(asdict(data_config))
    datalaoder = IndependentMixDataloader(data_config)
    databatch = datalaoder.get_databatch()
    print(databatch)
    
