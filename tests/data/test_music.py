import pytest

from markov_bridges.data.sequences.music_dataloaders import (
    LankhPianoRollDataloader,
)

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig
from markov_bridges.data.sequences.music_dataloaders import LankhPianoRollDataloader

def test_load_music():
    data_config = LakhPianoRollConfig()
    dataloader = LankhPianoRollDataloader(data_config)
    databatch = dataloader.get_databatch()
    
    #print(databatch.source_discrete.shape)
    #print(databatch.target_discrete.shape)

    databatch = next(dataloader.validation().__iter__())
    print(databatch.target_discrete)

if __name__=="__main__":
    test_load_music()
    