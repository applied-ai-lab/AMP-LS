import os

from core.models.vae_mdl import VAE
from core.components.logger import Logger
from core.utils.general_utils import AttrDict
from core.configs.default_data_configs.random import data_spec
from core.components.evaluator import ImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    "model": VAE,
    "logger": Logger,
    "evaluator": ImageEvaluator,
    "data_dir": ".",
    "num_epochs": 100,
    "batch_size": 16,
}
configuration = AttrDict(configuration)

model_config = {}

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec
