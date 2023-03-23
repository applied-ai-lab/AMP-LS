import os
import numpy as np
import torch.nn as nn

from core.utils.general_utils import AttrDict
from core.models.collision_kinematics_vae_mdl import CollisionKinematicsVAE
from core.models.kinematics_vae_mdl import KinematicsVAE
from core.planner.components.environment import GymEnv
from core.configs.model.collision_kinematics_vae.panda.conf import *


model_config.update(
    AttrDict(
        checkpt_path="./experiments/model/collision_kinematics_vae/panda/geco.g0.0002.z512.latent7.st16.6d.kcrl0.split.pr.iterD.scratch.1M.x1.5.wg.w20.shuffle"
    )
)

geco_params = {
    "goal1": 0.0061,
    "goal2": 0.0566,
    "geco_lambda1_init": 0.0,
    "geco_lambda2_init": 0.431,
    "step_size1": 0.195,
    "step_size2": 0.542,
    "geco_lambda1_min": 0.00035,
    "geco_lambda2_min": 0.01,
    "geco_lambda1_max": 24.5,
    "geco_lambda2_max": 1.3,
}
geco_params = AttrDict(geco_params)
