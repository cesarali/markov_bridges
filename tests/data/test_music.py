import pytest

from markov_bridges.data.music_dataloaders import (
    LankhPianoRollDataloader,
    get_data
)

from markov_bridges.configs.config_classes.data.music_configs import LakhPianoRollConfig

def test_load_music():
    data_config = LakhPianoRollConfig()
    train_data,  test_data, descramble_key = get_data(data_config)


if __name__=="__main__":
    test_load_music()
    