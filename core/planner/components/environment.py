from contextlib import contextmanager
from functools import partial
import torch
import numpy as np

# from torchvision.transforms import Resize
from PIL import Image
import cv2

from core.utils.general_utils import ParamDict, AttrDict, map_recursive
from core.utils.pytorch_utils import ar2ten, ten2ar


class BaseEnvironment:
    """Implements basic environment interface."""

    # TODO add frame skip interface

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass
        yield
        pass

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "device": None,  # device that all tensors should get transferred to
                "screen_width": 400,  # width of rendered images
                "screen_height": 400,  # height of rendered images
            }
        )
        return default_dict

    def reset(self):
        """Resets all internal variables of the environment."""
        raise NotImplementedError

    def step(self, action):
        """Performs one environment step. Returns dict <next observation, reward, done, info>."""
        raise NotImplementedError

    def render(self, mode="rgb_array"):
        """Renders current environment state. Mode {'rgb_array', 'none'}."""
        raise NotImplementedError

    def _wrap_observation(self, obs):
        """Process raw observation from the environment before return."""
        return obs

    @property
    def agent_params(self):
        """Parameters for agent that can be handed over after env is constructed."""
        return AttrDict()


class GymEnv(BaseEnvironment):
    """Wrapper around openai/gym environments."""

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)

        from mujoco_py.builder import MujocoException

        self._mj_except = MujocoException

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "name": None,  # name of openai/gym environment
                "reward_norm": 1.0,  # reward normalization factor
                "punish_reward": -100,  # reward used when action leads to simulation crash
                "unwrap_time": True,  # removes time limit wrapper from envs so that done is not set on timeout
                "env_params": AttrDict(),
            }
        )

        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        obs = self._env.reset()
        return self._wrap_observation(obs)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = ten2ar(action)
        try:
            obs, reward, done, info = self._env.step(action)
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = (
                self._hp.punish_reward
            )  # this avoids that the agent is going to these states again
            done = np.array(
                True
            )  # terminate episode (observation will get overwritten by env reset)
            info = {}

        return self._wrap_observation(obs), reward, np.array(done), info

    def render(self, mode="rgb_array"):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._env.render(mode=mode)
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = Resize((self._hp.screen_height, self._hp.screen_width))(
            Image.fromarray(img)
        )
        return np.array(img) / 255.0

    def _render(self, mode="rgb_array"):
        return self._env.render(mode=mode)

    def set_config(self, spec):
        try:
            self._env.set_config(spec)
        except AttributeError:
            pass
        self._spec = spec

    def get_dataset(self):
        dataset = None
        try:
            dataset = self._env.get_dataset()
            dataset.pop("timeouts")
            dataset["action"] = dataset.pop("actions")
            dataset["done"] = dataset.pop("terminals").astype(np.float32)
            dataset["observation"] = dataset.pop("observations")
            dataset["observation_next"] = np.concatenate(
                [
                    dataset["observation"][1:],
                    np.zeros_like(dataset["observation"][0])[np.newaxis, :],
                ],
                axis=0,
            )
            dataset["reward"] = dataset.pop("rewards")
        except AttributeError:
            pass
        return dataset

    def _make_env(self, id):
        """Instantiates the environment given the ID."""
        import gym
        from gym import wrappers

        env = gym.make(id, **self._hp.env_params)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def get_episode_info(self):
        """Allows to return logging info about latest episode (sindce last reset)."""
        if hasattr(self._env, "get_episode_info"):
            return self._env.get_episode_info()
        return AttrDict()


class GoalConditionedEnv(BaseEnvironment):
    def __init__(self):
        self.goal = None

    def sample_goal(self):
        raise NotImplementedError("Please implement this method in a subclass.")

    def reset(self):
        self.goal = self.sample_goal()

    def _wrap_observation(self, obs):
        return np.concatenate([obs, self.goal])
