import os
import numpy as np
import random
import json
import imp
import h5py
import argparse
import tqdm
import optuna
import wandb
import torch
from optuna.integration.wandb import WeightsAndBiasesCallback

from core.data.gazebo.scripts.amp_ls.evaluate import evaluate
from core.utils.general_utils import AttrDict
from core.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)
from core.models.collision_kinematics_vae_mdl import CollisionKinematicsVAE
from core.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def objective(trial, args, model, scene_config):
    geco_goal1 = trial.suggest_float(
        "geco_goal1", args.min_geco_goal1, args.max_geco_goal1, step=0.0001
    )
    geco_goal2 = trial.suggest_float(
        "geco_goal2", args.min_geco_goal2, args.max_geco_goal2, step=0.0001
    )
    geco_lambda2_init = trial.suggest_float(
        "geco_lambda_init2",
        args.min_geco_lambda_init2,
        args.max_geco_lambda_init2,
        step=0.001,
    )
    geco_step_size1 = trial.suggest_float(
        "geco_step_size1",
        args.min_geco_step_size1,
        args.max_geco_step_size1,
        step=0.001,
    )
    geco_step_size2 = trial.suggest_float(
        "geco_step_size2",
        args.min_geco_step_size2,
        args.max_geco_step_size2,
        step=0.001,
    )
    geco_lambda1_max = trial.suggest_float(
        "geco_lambda1_max",
        args.min_geco_lambda1_max,
        args.max_geco_lambda1_max,
        step=0.01,
    )
    geco_lambda2_max = trial.suggest_float(
        "geco_lambda2_max",
        args.min_geco_lambda2_max,
        args.max_geco_lambda2_max,
        step=0.01,
    )
    geco_lambda1_min = trial.suggest_float(
        "geco_lambda1_min",
        args.min_geco_lambda1_min,
        args.max_geco_lambda1_min,
        step=0.0001,
    )
    geco_lambda2_min = trial.suggest_float(
        "geco_lambda2_min",
        args.min_geco_lambda2_min,
        args.max_geco_lambda2_min,
        step=0.001,
    )
    alpha1 = trial.suggest_float("alpha1", args.min_alpha1, args.max_alpha1, step=0.05)
    alpha2 = trial.suggest_float("alpha2", args.min_alpha2, args.max_alpha2, step=0.05)

    geco_params = AttrDict(
        goal1=geco_goal1,
        goal2=geco_goal2,
        geco_lambda1_init=args.geco_lambda1_init,
        geco_lambda2_init=geco_lambda2_init,
        step_size1=geco_step_size1,
        step_size2=geco_step_size2,
        geco_lambda1_min=geco_lambda1_min,
        geco_lambda2_min=geco_lambda2_min,
        geco_lambda1_max=geco_lambda1_max,
        geco_lambda2_max=geco_lambda2_max,
        alpha1=alpha1,
        alpha2=alpha2,
    )

    success_rate = evaluate(args, model, geco_params, scene_config)
    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--rollout_len", type=int, default=150)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--latent_lr", type=float, default=0.06)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--rot_threshold", type=float, default=15)
    parser.add_argument("--min_geco_goal1", type=float, default=0.0)
    parser.add_argument("--max_geco_goal1", type=float, default=0.01)
    parser.add_argument("--geco_lambda1_init", type=float, default=0.0)
    parser.add_argument("--min_geco_lambda1_max", type=float, default=10)
    parser.add_argument("--max_geco_lambda1_max", type=float, default=30)
    parser.add_argument("--min_geco_lambda1_min", type=float, default=0.0001)
    parser.add_argument("--max_geco_lambda1_min", type=float, default=0.01)
    parser.add_argument("--min_geco_step_size1", type=float, default=0.1)
    parser.add_argument("--max_geco_step_size1", type=float, default=1.0)
    parser.add_argument("--min_alpha1", type=float, default=0.8)
    parser.add_argument("--max_alpha1", type=float, default=0.95)
    parser.add_argument("--min_geco_goal2", type=float, default=0.0)
    parser.add_argument("--max_geco_goal2", type=float, default=0.1)
    parser.add_argument("--min_geco_lambda_init2", type=float, default=0.0)
    parser.add_argument("--max_geco_lambda_init2", type=float, default=1.0)
    parser.add_argument("--min_geco_step_size2", type=float, default=0.1)
    parser.add_argument("--max_geco_step_size2", type=float, default=1.0)
    parser.add_argument("--min_geco_lambda2_max", type=float, default=0.5)
    parser.add_argument("--max_geco_lambda2_max", type=float, default=3)
    parser.add_argument("--min_geco_lambda2_min", type=float, default=0.001)
    parser.add_argument("--max_geco_lambda2_min", type=float, default=0.1)
    parser.add_argument("--min_alpha2", type=float, default=0.7)
    parser.add_argument("--max_alpha2", type=float, default=0.95)
    parser.add_argument("--traj_suffix", type=str, default="")
    parser.add_argument("--save_every_traj", action="store_true")
    parser.add_argument("--use_saved_pc", action="store_true")
    parser.add_argument(
        "--scene_config_path",
        default=os.path.join(
            os.environ["DATA_DIR"], "gazebo/env_configs/scene_config_010_trajs.json"
        ),
        help="init config json file.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--num_traj", type=int, default=None)
    parser.add_argument("--collision_threshold", type=float, default=0.4)
    parser.add_argument("--ori_scale", type=float, default=1.0)
    parser.add_argument("--max_joint_diff", type=float, default=0.05)
    parser.add_argument("--dont_save", action="store_false")
    parser.add_argument("--skip_explicit_collision", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    sampler = optuna.samplers.TPESampler

    wandb_kwargs = {"project": "lsmp", "entity": "a2i", "name": args.run_name}

    wandbc = WeightsAndBiasesCallback(
        metric_name="success_rate", wandb_kwargs=wandb_kwargs
    )

    conf_path = get_config_path(args.path)
    conf_module = imp.load_source("conf", conf_path)
    conf_model = conf_module.model_config
    conf_model.update(device=device)

    model = CollisionKinematicsVAE(conf_model).to(device)
    model.device = device

    weights_file = CheckpointHandler.get_resume_ckpt_file(
        "latest", os.path.join(conf_model.checkpt_path, "weights")
    )
    CheckpointHandler.load_weights(weights_file, model, load_step=False, load_opt=False)
    model.eval()
    args.checkpt_path = conf_model.checkpt_path
    with open(args.scene_config_path, "r") as fp:
        scene_config = json.loads(fp.read())

    def print_best_callback(study, trial):
        print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

    study = optuna.create_study(direction="maximize", sampler=sampler())
    study.optimize(
        lambda trial: objective(trial, args, model, scene_config),
        n_trials=1000,
        callbacks=[wandbc, print_best_callback],
    )

    f = "best_{}".format
    for param_name, param_value in study.best_trial.params.items():
        wandb.run.summary[f(param_name)] = param_value

    wandb.run.summary["best accuracy"] = study.best_trial.value

    wandb.log(
        {
            "optuna_optimization_history": optuna.visualization.plot_optimization_history(
                study
            ),
            "optuna_param_importances": optuna.visualization.plot_param_importances(
                study
            ),
        }
    )

    wandb.finish()
