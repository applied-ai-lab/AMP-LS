import os
import numpy as np
import torch.nn as nn

from core.utils.general_utils import AttrDict
from core.models.collision_kinematics_vae_mdl import (
    CollisionKinematicsVAE,
    CollisionKinematicsVAE,
)
from core.models.kinematics_vae_mdl import KinematicsVAE
from core.planner.agents.lsmp_agent import (
    LSMPAgent,
    PandaLSMPAgent,
    PandaSceneLSMPAgent,
)
from core.planner.components.environment import GymEnv

current_dir = os.path.dirname(os.path.realpath(__file__))

notes = "used to test the planner implementation"

configuration = {
    "seed": 123,
    "agent": PandaLSMPAgent,
    "environment": GymEnv,
    "data_dir": ".",
    "num_epochs": 1,
    "max_rollout_len": 500,
    "n_steps_per_epoch": 10000,
}
configuration = AttrDict(configuration)

model_params = {
    "state_dim": 16,
    "nz_enc": 512,
    "nz_mid": 512,
    "nz_vae": 7,
    "activation_fn": nn.ELU(inplace=True),
    "normalization": "none",
    "robot_urdf": "./core/data/scene_collision/assets/data/panda/panda.urdf",
}
model_params = AttrDict(model_params)

geco_params = {
    "geco_lambda1_init": 0.0822090976046263,
    "geco_lambda2_init": 0.0005177875263458441,
    "goal1": 0.00981557011630703,
    "goal2": 0.0034138207738205794,
    "step_size1": 0.1849586857453003,
    "step_size2": 0.6245755561006467,
    "geco_lambda1_max": 11.099661216254745,
    "geco_lambda2_max": 27.50152161955105,
}
geco_params = AttrDict(geco_params)

# Agent
agent_config = AttrDict(
    model=KinematicsVAE,
    model_params=model_params,
    model_checkpoint="./experiments/model/kinematics_vae/panda/geco.g0.0002.z512.latent7.st16.6d.kcrl0.split.pr.iterD",
    model_epoch="latest",
    batch_size=1,
    geco_params=geco_params,
    has_rot=True,
    latent_lr=0.04,
)

# Dataset - Random data
data_config = AttrDict()

# Environment
env_config = AttrDict(
    name="PandaReach-v0",
    env_params=AttrDict(
        config=AttrDict(
            moving_goal=True,
            goal=np.array(
                [0.6532234781115188, 0.15758841218199166, 0.3472696881268326]
            ),
            goal_quat=np.array(
                [
                    0.9199314497466583,
                    -0.3894288051548106,
                    -0.01114679087825901,
                    -0.044125758189799295,
                ]
            )
            # goal_quat=np.array([9.3610537e-01, 3.5150602e-01, 2.8480589e-04, 1.2257536e-02]),
            # use_tag_goal=True,
            # tag_id='tag_0'
        )
    ),
)
