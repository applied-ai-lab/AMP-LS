import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from functools import partial
from torch.optim import Adam, SGD

from core.utils.general_utils import (
    ParamDict,
    get_clipped_optimizer,
    AttrDict,
    prefix_dict,
    map_dict,
    nan_hook,
    np2obj,
    ConstantSchedule,
)
from core.utils.pytorch_utils import RAdam, remove_grads, map2np, map2torch
from core.utils.vis_utils import add_caption_to_img, add_captions_to_seq
from core.planner.components.normalization import DummyNormalizer
from core.planner.components.policy import Policy
from core.components.checkpointer import CheckpointHandler


class BaseAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)
        self.device = self._hp.device
        self._is_train = True  # indicates whether agent should sample in training mode
        self._rand_act_mode = (
            False  # indicates whether agent should act randomly (for warmup collection)
        )
        self._rollout_mode = False  # indicates whether agent is run in rollout mode (omit certain policy outputs)
        self._obs_normalizer = self._hp.obs_normalizer(self._hp.obs_normalizer_params)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "device": None,  # pytorch device
                "discount_factor": 0.99,  # discount factor for RL update
                "optimizer": "adam",  # supported: 'adam', 'radam', 'rmsprop', 'sgd'
                "adam_eps": 1e-8,
                "gradient_clip": None,  # max grad norm, if None no clipping
                "momentum": 0,  # momentum in RMSProp / SGD optimizer
                "adam_beta": 0.9,  # beta1 param in Adam
                "update_iterations": 1,  # number of iteration steps per one call to 'update(...)'
                "target_network_update_factor": 5e-3,  # percentage of new weights that are carried over
                "batch_size": 64,  # size of the experience batch used for updates
                "obs_normalizer": DummyNormalizer,  # observation normalization class
                "obs_normalizer_params": {},  # parameters for optimization norm class
                "log_videos": True,  # whether to log videos during logging
                "log_video_caption": False,  # whether to add captions to video
            }
        )
        return default_dict

    def act(self, obs):
        """Returns policy output dict given observation (random action if self._rand_act_mode is set)."""
        if self._rand_act_mode:
            return self._act_rand(obs)
        else:
            return self._act(obs)

    def _act(self, obs):
        """Implements act method in child class."""
        raise NotImplementedError

    def _act_rand(self, obs):
        """Returns random action with proper dimension. Implemented in child class."""
        raise NotImplementedError

    def update(self, experience_batch):
        """Updates the policy given a batch of experience."""
        raise NotImplementedError

    def add_experience(self, experience_batch):
        """Provides interface for adding additional experience to agent replay, needs to be overwritten by child."""
        print("### This agent does not support additional experience! ###")

    def log_outputs(self, logging_stats, rollout_storage, logger, log_images, step):
        """Visualizes/logs all training outputs."""
        logger.log_scalar_dict(
            logging_stats, prefix="train" if self._is_train else "val", step=step
        )

        if log_images:
            assert rollout_storage is not None  # need rollout data for image logging
            # log rollout videos with info captions
            if "image" in rollout_storage and self._hp.log_videos:
                if self._hp.log_video_caption:
                    vids = [
                        np.stack(
                            add_captions_to_seq(rollout.image, np2obj(rollout.info))
                        ).transpose(0, 3, 1, 2)
                        for rollout in rollout_storage.get()[-logger.n_logged_samples :]
                    ]
                else:
                    vids = [
                        np.stack(rollout.image).transpose(0, 3, 1, 2)
                        for rollout in rollout_storage.get()[-logger.n_logged_samples :]
                    ]
                logger.log_videos(vids, name="rollouts", step=step)
            self.visualize(logger, rollout_storage, step)

    def visualize(self, logger, rollout_storage, step):
        """Optionally allows to further visualize the internal state of agent (e.g. replay buffer etc.)"""
        pass

    def reset(self):
        """Can be used for any initializations of agent's state at beginning of episode."""
        pass

    def save_state(self, save_dir):
        """Provides interface to save any internal state variables (like replay buffers) to disk."""
        pass

    def load_state(self, save_dir):
        """Provides interface to load any internal state variables (like replay buffers) from disk."""
        pass

    def sync_networks(self):
        """Syncs network parameters across workers."""
        raise NotImplementedError

    def _copy_to_target_network(self, target, source):
        """Completely copies weights from source to target."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def _get_optimizer(self, optimizer, model, lr):
        """Returns an instance of the specified optimizers on the parameters of the model with specified learning rate."""
        if optimizer == "adam":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=Adam,
                betas=(self._hp.adam_beta, 0.999),
                eps=self._hp.adam_eps,
            )
        elif optimizer == "radam":
            get_optim = partial(
                get_clipped_optimizer,
                optimizer_type=RAdam,
                betas=(self._hp.adam_beta, 0.999),
            )
        elif optimizer == "sgd":
            get_optim = partial(
                get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum
            )
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optimizer))
        optim = partial(get_optim, gradient_clip=self._hp.gradient_clip)
        return optim(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def _perform_update(self, loss, opt, network, retain_graph=False):
        """Performs one backward gradient step on the loss using the given optimizer. Also syncs gradients."""
        nan_hook(loss)
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self._hp.num_workers > 1:
            sync_grads(network)

        grads = [p.grad for p in network.parameters()]
        nan_hook(grads)

        opt.step()

    @staticmethod
    def load_model_weights(model, checkpoint, epoch="latest"):
        """Loads weights for a given model from the given checkpoint directory."""
        checkpoint_dir = (
            checkpoint
            if os.path.basename(checkpoint) == "weights"
            else os.path.join(checkpoint, "weights")
        )  # checkpts in 'weights' dir
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model=model)

    @staticmethod
    def _remove_batch(d):
        """Adds batch dimension to all tensors in d."""
        return map_dict(
            lambda x: x[0]
            if (isinstance(x, torch.Tensor) or isinstance(x, np.ndarray))
            else x,
            d,
        )

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with agent.val_mode(): ...<do something>..."""
        self._is_train = False
        self.call_children("switch_to_val", Policy)
        yield
        self._is_train = True
        self.call_children("switch_to_train", Policy)

    @contextmanager
    def rand_act_mode(self):
        """Performs random actions within context. To be used like: with agent.rand_act_mode(): ...<do something>..."""
        self._rand_act_mode = True
        yield
        self._rand_act_mode = False

    @contextmanager
    def rollout_mode(self):
        """Sets rollout parameters if desired."""
        self._rollout_mode = True
        self.call_children("switch_to_rollout", Policy)
        yield
        self._rollout_mode = False
        self.call_children("switch_to_non_rollout", Policy)

    def call_children(self, fn, cls):
        """Call function with name fn in all submodules of class cls."""

        def conditional_fn(module):
            if isinstance(module, cls):
                getattr(module, fn).__call__()

        self.apply(conditional_fn)
