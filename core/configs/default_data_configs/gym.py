import os, sys

from core.utils.general_utils import AttrDict
from core.data.gym.gym_data_loader import GymOfflineDataset

dataset_spec = AttrDict(
    dataset_class=GymOfflineDataset,
    discount_factor=0.99,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    resolution=64,
)
