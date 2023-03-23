import os, sys
import numpy as np

from core.utils.general_utils import AttrDict

data_spec = AttrDict(
    dataset_class=None,
    discount_factor=0.99,
    obs_max=5,
    split_frac=0.9,
    resolution=64,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    max_seq_len=500,
    crop_rand_subseq=True,
    subseq_len=2,
    n_obj_low=10,
    n_obj_high=20,
    has_pos=True,
    table_dims=np.array([1.0, 1.6, 0.5]),
)
