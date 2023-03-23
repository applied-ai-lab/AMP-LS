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
    "agent": PandaSceneLSMPAgent,
    "environment": GymEnv,
    "data_dir": ".",
    "num_epochs": 1,
    "max_rollout_len": 2000,
    "n_steps_per_epoch": 10000,
}
configuration = AttrDict(configuration)

model_params = {
    "has_rot": True,
    "state_dim": 16,
    "nz_enc": 512,
    "nz_mid": 512,
    "nz_vae": 7,
    "activation_fn": nn.ELU(inplace=True),
    "normalization": "none",
    "bounds": [[-0.5, -0.8, 0.24], [0.5, 0.8, 0.6]],
    "relative_geometry": False,
}
model_params = AttrDict(model_params)
geco_params = {
    "goal1": 0.0011,
    "goal2": 0.0268,
    "geco_lambda1_init": 0.004,
    "geco_lambda2_init": 4.493,
    "step_size1": 0.18,
    "step_size2": 2.324,
    "geco_lambda1_max": 19.69,
    "geco_lambda2_max": 0.52,
    "geco_lambda1_min": 1e-10,
    "geco_lambda2_min": 0.01803,
    "alpha1": 0.9,
    "alpha2": 0.8,
}
geco_params = AttrDict(geco_params)

# Agent
agent_config = AttrDict(
    model=CollisionKinematicsVAE,
    model_params=model_params,
    model_checkpoint="./experiments/model/collision_kinematics_vae/panda/geco.g0.0002.z512.latent7.st16.6d.kcrl0.split.pr.iterD.scratch.1M.x1.5.wg.w20.shuffle",
    model_epoch="latest",
    batch_size=1,
    geco_params=geco_params,
    has_rot=True,
    aggregate_pc=False,
    online_planning=True,
    collision_threshold=0.2,
    latent_lr=0.06,
    ori_scale=1.1,
)

# Dataset - Random data
data_config = AttrDict()

# Environment
env_config = AttrDict(
    # name='PandaReach-v0',
    # name='PandaTableReach-v0',
    name="PandaConveyorReach-v0",
    env_params=AttrDict(
        config=AttrDict(
            use_tag_goal=False,
            tag_id="tag_0",
            # dt=0.2
            dt=0.1,
        )
    ),
)
