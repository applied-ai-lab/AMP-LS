import os
import imp
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch

from core.planner.components.params import get_args
from core.train import (
    set_seeds,
    make_path,
    datetime_str,
    save_config,
    get_exp_dir,
    save_checkpoint,
)
from core.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)
from core.utils.general_utils import (
    AttrDict,
    ParamDict,
    AverageTimer,
    timing,
    pretty_print,
    listdict2dictlist,
)
from core.planner.utils.wandb import WandBLogger
from core.planner.components.sampler import Sampler
from core.planner.components.rollout_storage import RolloutStorage
import core.planner.envs

WANDB_PROJECT_NAME = "amp_ls"
WANDB_ENTITY_NAME = "a2i"


class Planner:
    """Sets up RL training loop, instantiates all components, runs training."""

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        self.conf.group = args.group
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(
            self.conf.exp_dir, args.path, args.prefix, args.new_dir
        )
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, "log")
        print("using log dir: ", log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1:
            self._hp.seed = args.seed  # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        # set up logging
        print("Running base worker.")
        self.logger = self.setup_logging(self.conf, self.log_dir)

        # build env
        self.conf.env.seed = self._hp.seed
        self.env = self._hp.environment(self.conf.env)
        self.conf.agent.env_params = (
            self.env.agent_params
        )  # (optional) set params from env for agent
        if self.is_chef:
            pretty_print(self.conf)

        # build agent
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        self.sampler = self._hp.sampler(
            self.conf.sampler,
            self.env,
            self.agent,
            self.logger,
            self._hp.max_rollout_len,
        )

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 0  # no warmup if we reload from checkpoint!

        # start training/evaluation
        self.plan()

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "seed": None,
                "agent": None,
                "data_dir": None,  # directory where dataset is in
                "environment": None,
                "sampler": Sampler,  # sampler type used
                "exp_path": None,  # Path to the folder with experiments
                "num_epochs": 200,
                "max_rollout_len": 1000,  # maximum length of the performed rollout
                "logging_target": "wandb",  # where to log results to
            }
        )
        return default_dict

    def plan(self):
        """Evaluate agent."""
        val_rollout_storage = RolloutStorage()
        episode_info_list = []
        # with self.agent.val_mode():
        with timing("Eval rollout time: "):
            for _ in range(
                self.args.n_val_samples
            ):  # for efficiency instead of self.args.n_val_samples
                val_rollout_storage.append(
                    self.sampler.sample_episode(is_train=False, render=True)
                )
                episode_info_list.append(self.sampler.get_episode_info())

        # need this because the agents becomes the training mode after calling sample_episode()
        with self.agent.val_mode():
            rollout_stats = val_rollout_storage.rollout_stats()
            episode_info_dict = listdict2dictlist(episode_info_list)
            for key in episode_info_dict:
                episode_info_dict[key] = np.mean(episode_info_dict[key])
            rollout_stats.update(episode_info_dict)
            if self.is_chef:
                with timing("Eval log time: "):
                    self.agent.log_outputs(
                        rollout_stats,
                        val_rollout_storage,
                        self.logger,
                        log_images=True,
                        step=self.global_step,
                    )
            del val_rollout_storage

    def load_rollouts(self):
        self.conf.data.device = self.device.type
        rollouts = self.env.get_dataset()
        if rollouts is None:
            rollouts = self.conf.data.dataset_spec.dataset_class(
                self.conf.data.dataset_spec.data_dir,
                self.conf.data,
                resolution=self.conf.data.dataset_spec.resolution,
                phase=None,
            ).data_dict
        return rollouts

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and agent configs
        print("loading from the config file {}".format(conf.conf_path))
        conf_module = imp.load_source("conf", conf.conf_path)
        conf.general = conf_module.configuration
        conf.agent = conf_module.agent_config
        conf.agent.device = self.device

        # data config
        conf.data = conf_module.data_config

        # environment config
        conf.env = conf_module.env_config
        conf.env.device = (
            self.device
        )  # add device to env config as it directly returns tensors

        # sampler config
        conf.sampler = (
            conf_module.sampler_config
            if hasattr(conf_module, "sampler_config")
            else AttrDict({})
        )
        conf.sampler.device = self.device

        # model loading config
        conf.ckpt_path = (
            conf.agent.checkpt_path if "checkpt_path" in conf.agent else None
        )

        # load notes if there are any
        if self.args.notes != "":
            conf.notes = self.args.notes
        else:
            try:
                conf.notes = conf_module.notes
            except:
                conf.notes = ""

        # load config overwrites
        if self.args.config_override != "":
            for override in self.args.config_override.split(","):
                key_str, value_str = override.split("=")
                keys = key_str.split(".")
                curr = conf
                for key in keys[:-1]:
                    curr = curr[key]
                curr[keys[-1]] = type(curr[keys[-1]])(value_str)

        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print("Writing to the experiment directory: {}".format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(
                conf.conf_path,
                os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"),
            )

            # setup logger
            logger = None
            if (
                self.args.mode == "train"
                or self.args.mode == "val"
                or self.args.mode == "traj_opt"
            ):
                ".".join(self.args.path.split("/")[2:])
                exp_name = (
                    f"{'.'.join(self.args.path.split('/')[2:])}_{self.args.prefix}"
                    if self.args.prefix
                    else ".".join(self.args.path.split("/")[2:])
                )
                if self._hp.logging_target == "wandb":
                    logger = WandBLogger(
                        exp_name,
                        WANDB_PROJECT_NAME,
                        entity=WANDB_ENTITY_NAME,
                        path=self._hp.exp_path,
                        conf=conf,
                    )
                else:
                    raise NotImplementedError  # TODO implement alternative logging (e.g. TB)
            return logger

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def resume(self, ckpt, path=None):
        path = (
            os.path.join(self._hp.exp_path, "weights")
            if path is None
            else os.path.join(path, "weights")
        )
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        # TODO(karl): check whether that actually loads the optimizer too
        self.global_step, start_epoch, _ = CheckpointHandler.load_weights(
            weights_file,
            self.agent,
            load_step=True,
            strict=self.args.strict_weight_loading,
        )
        self.agent.load_state(self._hp.exp_path)
        self.agent.to(self.device)
        return start_epoch


if __name__ == "__main__":
    Planner(args=get_args())
