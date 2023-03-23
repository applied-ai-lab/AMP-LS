from gym.envs.registration import register
from core.utils.general_utils import AttrDict

register(
    id="GazeboPanda-v0",
    entry_point="core.rl.envs.gazebo_panda:GazeboPandaEnv",
    kwargs={"config": AttrDict()},
)

register(
    id="GazeboPandaBlock-v0",
    entry_point="core.rl.envs.gazebo_panda:GazeboPandaBlockEnv",
    kwargs={"config": AttrDict()},
)
register(
    id="GazeboPandaConveyor-v0",
    entry_point="core.rl.envs.gazebo_panda:GazeboPandaConveyorEnv",
    kwargs={"config": AttrDict()},
)

register(
    id="PandaReach-v0",
    entry_point="core.rl.envs.panda:PandaReachEnv",
    kwargs={"config": AttrDict()},
)

register(
    id="PandaTableReach-v0",
    entry_point="core.rl.envs.panda:PandaTableReachEnv",
    kwargs={"config": AttrDict()},
)

register(
    id="PandaConveyorReach-v0",
    entry_point="core.rl.envs.panda:PandaConveyorReachEnv",
    kwargs={"config": AttrDict()},
)
