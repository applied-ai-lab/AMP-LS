import os

from core.models.kinematics_vae_mdl import KinematicsVAE
from core.components.logger import Logger
from core.utils.general_utils import AttrDict
from core.configs.default_data_configs.scene_collision import data_spec
from core.data.scene_collision.scene_collision_data_loader import (
    RobotCollisionFreeJointDataset,
)
from core.components.evaluator import (
    ImageEvaluator,
    DummyEvaluator,
    MultiImageEvaluator,
    RotationEvaluator,
)
from core.utils.aug_utils import *


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    "model": KinematicsVAE,
    "model_test": KinematicsVAE,
    "data_dir": os.path.join(os.environ["DATA_DIR"], "./panda/joints"),
    "logger": Logger,
    "logger_test": Logger,
    "evaluator": RotationEvaluator,  # DummyEvaluator
    "num_epochs": 200,
    "epoch_cycles_train": 40,
    "batch_size": 256,
    "lr": 1e-4,
    "optimizer": "adam",
}
configuration = AttrDict(configuration)

model_config = {
    "has_rot": True,
    "state_dim": 16,
    "nz_enc": 512,
    "nz_mid": 512,
    "nz_vae": 7,
    "activation_fn": nn.ELU(inplace=True),
    "normalization": "none",
    "fixed_beta": 1,
    "kinematics_consistency_mse_weight": 0.0,
    "kinematics_rot_consistency_mse_weight": 0.0,
    "rot_loss_weight": 1,
    "geco_params": AttrDict(
        geco_lambda_init=1,
        geco_lambda_min=1e-10,
        geco_lambda_max=1e6,
        goal=0.0002,
        step_size=1e-1,
        speedup=10,
    ),
}

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class = RobotCollisionFreeJointDataset
data_config.dataset_spec.has_rot = model_config["has_rot"]
data_config.dataset_spec.subseq_len = 1
data_config.dataset_spec.robot_urdf = (
    "./core/data/scene_collision/assets/data/panda/panda.urdf"
)
data_config.data_dir = os.path.join(os.environ["DATA_DIR"], "./panda/joints")
