import os
import numpy as np
import random
import json
import h5py
import copy
import argparse
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
import imp
from typing import List, Sequence
import trimesh
import trimesh.transformations as tra

from core.utils.eval_utils import mean_angle_btw_vectors
from core.models.collision_kinematics_vae_mdl import CollisionKinematicsVAE
from core.utils.general_utils import AttrDict
from core.utils.pytorch_utils import ten2ar, map2torch, ar2ten, RAdam
from core.utils.transform_utils import mat2quat, quat2mat
from core.modules.geco_modules import GECO, GECO2

# from trac_ik_python.trac_ik import IK
from core.components.checkpointer import (
    CheckpointHandler,
    save_cmd,
    save_git,
    get_config_path,
)
from core.utils.transform_utils import ortho6d2mat, mat2euler

# from core.rl.envs.gazebo_panda import GazeboPandaEnv, GazeboPandaBlockEnv, GazeboPandaTestEnv, GazeboPandaConveyorEnv
from urdfpy import URDF
from core.data.scene_collision.src.robot import Robot

# use one of the following launch files for evaluation:
# roslaunch panda_moveit_config shapenet_empty_camera_connected.launch (no Panda in Gazebo)
# roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch (with Panda in Gazebo)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
asset_path = "./core/data/scene_collision/assets/"
urdf_path = os.path.join(asset_path, "./data/panda/panda.urdf")
robot = URDF.load(urdf_path)
panda_robot = Robot(urdf_path, "panda_link8", device=device)
# ik = IK('panda_link0', 'panda_link8', solve_type='Manipulation2', urdf_string=urdf_path)
links = [link for link in robot.links if link.collision_mesh is not None]


def format_log(
    loss_prior,
    prior_lambda,
    dist_loss,
    ori_loss,
    geco_lambda1,
    collision_loss,
    geco_lambda2,
    collision_prob,
    total_loss,
    dist,
    angle,
):
    return {
        "01. prior_loss": ten2ar(loss_prior).item(),
        "02. prior_lambda": prior_lambda,
        "03. dist_loss": ten2ar(dist_loss).item(),
        "04. ori_loss": ten2ar(ori_loss).item(),
        "05. target_lambda": ten2ar(geco_lambda1).item(),
        "06. collision_loss": ten2ar(collision_loss).item(),
        "07. collision_lambda": ten2ar(geco_lambda2).item(),
        "08. collision_prob": ten2ar(collision_prob).item(),
        "09. total_loss": ten2ar(total_loss).item(),
        "10. dist diff": dist,
        "11. angle diff": angle,
    }


def _get_rotated(R):
    INIT_AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if len(R.shape) < 3:
        R = np.expand_dims(R, axis=0)
    return np.transpose(np.matmul(R, np.transpose(INIT_AXES)), axes=[0, 2, 1])


def compute_dist_loss(decoded_output, target):
    return torch.dist(decoded_output, target.detach())


def compute_path_length(ee_xyz):
    prev_ee_pos = None
    distance = 0.0
    for ee_pos in ee_xyz:
        if prev_ee_pos is not None:
            distance += np.linalg.norm(prev_ee_pos - ee_pos)
        prev_ee_pos = ee_pos
    return distance


def update_inputs(inputs, link_pose, robot_to_model):
    trans = []
    rots = []
    # for i, link in enumerate(links[1:]):
    for i, link in enumerate(reversed(links[1:])):
        pose = robot_to_model @ link_pose[link]
        trans.append(pose[:3, 3])
        rots.append(pose[:3, :2].reshape(-1))
    trans = np.array(trans)[None]
    rots = np.array(rots)[None]
    inputs.update(map2torch(AttrDict(trans=trans[None], rots=rots[None]), device))
    return inputs


def obs_to_array(obs):
    obs["camera_pose"] = np.array(obs["camera_pose"])
    obs["robot_to_model"] = np.array(obs["robot_to_model"])
    obs["model_to_robot"] = np.array(obs["model_to_robot"])
    obs["robot_states"] = np.array(obs["robot_states"])
    obs["goal"] = np.array(obs["goal"])
    obs["goal_ori"] = np.array(obs["goal_ori"])
    obs["goal_joints"] = np.array(obs["goal_joints"])
    obs["pc"] = np.array(obs["pc"])
    obs["pc_label"] = np.array(obs["pc_label"])
    # obs['depth_img'] = np.array(obs['depth_img'])

    return obs


def aggregate_pc(cur_pc, new_pc, scene_pc_mask, camera_pose):
    # Filter tiny clusters of points and split into vis/occluding
    cam_model_tr = torch.from_numpy(camera_pose[:3, 3]).float().to(device)
    new_pc = torch.from_numpy(new_pc).float().to(device)
    vis_mask = torch.from_numpy(scene_pc_mask).to(device)
    dists = torch.norm(new_pc - cam_model_tr, dim=1) ** 2
    dists /= dists.max()
    nearest = torch.zeros_like(vis_mask)
    # occ_scene_pc = new_pc[~vis_mask & (nearest < 0.1 * dists)]
    scene_pc = new_pc[vis_mask & (nearest < 0.1 * dists)]

    if cur_pc is not None:
        concat_pc = np.concatenate([ten2ar(new_pc), ten2ar(cur_pc)], axis=0)
        indices = np.arange(len(concat_pc))
        selected_indices = np.random.choice(indices, size=len(new_pc))
        scene_pc = concat_pc[selected_indices]
        scene_pc = torch.from_numpy(scene_pc).float().to(device)
        return scene_pc
    else:
        return scene_pc


def prepare_pc(obs, bounds):
    label_map = obs["label_map"]
    scene_labels = obs["pc_label"]
    scene_pc_mask = np.logical_and(
        scene_labels != label_map["robot"], scene_labels != label_map["target"]
    )
    camera_pose = obs["camera_pose"]
    scene_pc = tra.transform_points(obs["pc"], camera_pose)
    scene_pc = obs["pc"]

    cur_scene_pc = aggregate_pc(None, scene_pc, scene_pc_mask, camera_pose)

    model_scene_pc = (
        ar2ten(obs["robot_to_model"].astype(np.float32), device)
        @ torch.cat(
            [
                cur_scene_pc,
                torch.ones((len(cur_scene_pc), 1)).to(device),
            ],
            axis=1,
        ).T
    )
    model_scene_pc = model_scene_pc[:3].T

    # model_scene_pc = ar2ten(model_scene_pc.astype(np.float32), self._hp.device)
    # Clip points to model bounds and feed in for features
    in_bounds = (model_scene_pc[..., :3] > bounds[0] + 1e-5).all(dim=-1)
    in_bounds &= (model_scene_pc[..., :3] < bounds[1] - 1e-5).all(dim=-1)
    return model_scene_pc[in_bounds]


def collision_between_joint_states(
    prev_joint,
    current_joint,
    collision_threshold,
    pc,
    obs,
    model,
    scene_features,
    samples=15,
):
    current_joint = ten2ar(current_joint)
    prev_joint = ten2ar(prev_joint)

    diff = (current_joint - prev_joint) / (samples + 1)
    joint_states = (
        prev_joint[None].repeat(samples, 0)
        + diff[None].repeat(samples, 0) * np.arange(1, samples + 1)[:, None]
    )
    panda_robot.set_joint_cfg(
        ar2ten(
            np.concatenate(
                [
                    joint_states.astype(np.float32),
                    np.ones((samples, 1)).astype(np.float32) * 0.04,
                ],
                axis=1,
            ),
            device,
        )
    )

    ee_pose = panda_robot.ee_pose
    ee_pos = ee_pose[:, :3, 3]
    ee_6d = ee_pose[:, :3, :2].permute((0, 2, 1)).reshape(len(ee_pose), -1)

    inputs = AttrDict(
        states=torch.cat(
            [ar2ten(joint_states.astype(np.float32), device), ee_pos, ee_6d], dim=-1
        ).unsqueeze(0),
        scene_pc=pc[None].unsqueeze(0).repeat(1, samples, 1, 1),
        scene_features=scene_features.detach(),
    )
    trans = []
    rots = []
    robot_to_model = ar2ten(obs["robot_to_model"].astype(np.float32), device)
    for i, link in enumerate(reversed(panda_robot.mesh_links[1:])):
        pose = robot_to_model @ panda_robot.link_poses[link]
        trans.append(pose[:, :3, 3])
        rots.append(pose[:, :3, :2].reshape(len(pose), -1))
    trans = torch.stack(trans).permute(1, 0, 2)
    rots = torch.stack(rots).permute(1, 0, 2)
    inputs.update(
        AttrDict(
            trans=trans[None],
            rots=rots[None],
        )
    )
    # inputs = map2torch(inputs, device=device)
    with torch.no_grad():
        encoded_output = model.encode(inputs)
        z = encoded_output.q.mu
        output = model.decode(z, inputs)
    collision_prob = torch.sigmoid(output.collision)
    if torch.sum(collision_prob > collision_threshold) > 0:
        indices = torch.nonzero(collision_prob > collision_threshold)
        return True, (ten2ar(indices[0][1] + 1)) / samples
    else:
        return False, 1


def interpolate_path(prev_joint, current_joint, timesteps):
    diff = (current_joint - prev_joint) / (timesteps + 1)
    intervals = diff * np.arange(1, timesteps + 1)[:, None]
    inter_joints = prev_joint[None].repeat(len(intervals), 0) + intervals
    return inter_joints


def evaluate(args, model, geco_params, scene_config=None):
    if scene_config is not None:
        num_traj = len(scene_config.keys())

    if args.num_traj is not None:
        num_traj = args.num_traj
    if not args.use_saved_pc:
        env_params = AttrDict(
            robot_client_params=AttrDict(use_fake_controller=True),
        )
        env = GazeboPandaEnv(env_params)

    trajs_dir = os.path.join(args.checkpt_path, "trajs")
    if not os.path.exists(trajs_dir):
        os.makedirs(trajs_dir)

    # load json file if exist
    filename = "lsmp" + f"_{num_traj}_trajs"
    output_filename = filename + "_" + args.traj_suffix + ".json"
    save_json_path = os.path.join(trajs_dir, output_filename)

    output = {}
    trajectories = {}
    planning_times = []
    path_lengths = []
    normalised_path_lengths = []
    success_ids = []
    dists = []
    angles = []

    i = 0
    for i in tqdm.tqdm(range(num_traj)):
        # print(f"Planning Trajectory {i + 1}/{num_traj}...")

        if args.use_saved_pc:
            obs = scene_config["%03d" % i]["obs"]
            obs = obs_to_array(obs)
        else:
            obs = env.reset_to(scene_config["%03d" % i])
            # obs = env.reset()
        pc = prepare_pc(obs, model.bounds)

        geco = GECO2(geco_params)

        ee_pose = robot.link_fk(obs["robot_states"], link="panda_link8")
        ee_pos = ee_pose[:3, 3]
        ee_6d = ee_pose[:3, :2].transpose().reshape(-1)

        states = np.concatenate([obs["robot_states"][:7], ee_pos, ee_6d], axis=-1)[None]

        link_pose = robot.link_fk(obs["robot_states"])

        inputs = AttrDict(
            states=states.reshape((1, 1, states.shape[-1])),
            scene_pc=pc[None].unsqueeze(0),
        )
        inputs = update_inputs(inputs, link_pose, obs["robot_to_model"])
        inputs = map2torch(inputs, device=device)
        scene_features = model.get_scene_features(inputs)
        inputs.update(AttrDict(scene_features=scene_features.detach()))

        with torch.no_grad():
            encoded_output = model.encode(inputs)
            z = encoded_output.q.mu.detach()
        feature = nn.Parameter(z)
        opt = Adam([feature], lr=args.latent_lr)

        # save planned path and scene info to trajectories output
        trajectories["%03d" % i] = {}

        trajectories["%03d" % i]["ee_jpos"] = [
            (obs["robot_states"][:7] / np.pi * 180.0).tolist()
        ]
        trajectories["%03d" % i]["pred_ee_xyz"] = [[0.0, 0.0, 0.0]]
        trajectories["%03d" % i]["target_xyz"] = obs["goal"].tolist()
        trajectories["%03d" % i]["scene_info"] = scene_config["%03d" % i]["scene_info"]
        trajectories["%03d" % i]["per_frame_info"] = [
            {
                "01. prior_loss": -1,
                "02. prior_lambda": 1,
                "03. dist_loss": -1,
                "04. ori_loss": -1,
                "05. dist_lambda": geco.geco_lambda1.cpu().detach().item(),
                "06. collision_loss": -1,
                "07. collision_lambda": geco.geco_lambda2.cpu().detach().item()
                if hasattr(geco, "geco_lambda3")
                else -1.0,
                "08. collision_prob": -1,
                "09. total_loss": -1,
                "10. dist diff": -1,
                "11. angle diff": -1,
            }
        ]

        collision = False
        planning_time = 0
        ee_xyz = []  # for computing path length
        dist_init_ee_to_goal = np.linalg.norm(
            ee_pos - obs["goal"]
        )  # for computing normalised path length

        decoded_output = model.decode(feature, inputs)
        current_joint = decoded_output.states.squeeze(0)[:, :7]
        current_ee = decoded_output.states.squeeze(0)[:, 7:10]
        trajectories["%03d" % i]["ee_jpos"].append(
            (ten2ar(current_joint[0]) / np.pi * 180.0).tolist()
        )
        trajectories["%03d" % i]["pred_ee_xyz"].append(ten2ar(current_ee[0]).tolist())

        goal_rot = quat2mat(obs["goal_ori"]).astype(np.float32)
        goal_ori = ar2ten(goal_rot[:3, :2].transpose().reshape(-1), device).unsqueeze(0)
        goal_ee = ar2ten(obs["goal"].astype(np.float32), device).unsqueeze(0)
        collision_target = torch.zeros_like(decoded_output.collision)
        prior_target = torch.zeros_like(feature)
        prev_joint = ten2ar(current_joint[0])
        prev_latent = nn.Parameter(feature.clone())
        prev_inputs = copy.deepcopy(inputs)
        prev_geco_state_dict = geco.state_dict()
        trials = 0
        colls = 0
        steps = 0
        adjust_ratio = False
        for _ in range(args.rollout_len):
            opt.zero_grad()

            current_joint = decoded_output.states.squeeze(0)[:, :7]
            current_ee = decoded_output.states.squeeze(0)[:, 7:10]
            current_ori = decoded_output.states.squeeze(0)[:, 10:]

            if torch.sigmoid(decoded_output.collision) > args.collision_threshold:
                prev_valid_joint = False
            else:
                prev_valid_joint = True
                prev_joint = ten2ar(current_joint[0])
                prev_ee_pos = ten2ar(current_ee[0])
                prev_latent = nn.Parameter(feature.clone().detach())
                prev_inputs = copy.deepcopy(inputs)
                prev_geco_state_dict = geco.state_dict()

            t0 = time.process_time()
            dist_loss = compute_dist_loss(current_ee, goal_ee)
            ori_loss = nn.MSELoss()(current_ori, goal_ori)
            collision_loss = torch.binary_cross_entropy_with_logits(
                decoded_output.collision, collision_target
            ).mean()
            loss_prior = F.mse_loss(feature, prior_target)
            target_loss = dist_loss + ori_loss
            loss = geco.loss(loss_prior, target_loss, collision_loss)
            loss.backward()
            opt.step()

            t1 = time.process_time()
            planning_time += t1 - t0

            current_joint = ten2ar(model.decode_states(feature).squeeze(0)[0])[:7]
            link_pose = robot.link_fk(np.concatenate([current_joint, np.array([0.04])]))
            ee_pose = link_pose[robot._link_map["panda_link8"]]
            ee_pos = ee_pose[:3, 3]
            ee_rot = ee_pose[:3, :3]
            dist = np.linalg.norm(obs["goal"] - ee_pos)
            angle = mean_angle_btw_vectors(_get_rotated(ee_rot), _get_rotated(goal_rot))

            if dist < args.threshold and angle < args.rot_threshold:
                per_frame_info = format_log(
                    loss_prior,
                    1.0,
                    dist_loss,
                    ori_loss,
                    geco.geco_lambda1,
                    collision_loss,
                    geco.geco_lambda2,
                    torch.sigmoid(decoded_output.collision),
                    loss,
                    dist,
                    angle,
                )
                trajectories["%03d" % i]["per_frame_info"].append(per_frame_info)
                # print("success")
                ee_xyz.append(ee_pos)
                trajectories["%03d" % i]["ee_jpos"].append(
                    (current_joint / np.pi * 180.0).tolist()
                )
                trajectories["%03d" % i]["pred_ee_xyz"].append(ee_pos.tolist())
                planning_times.append(planning_time)

                path_length = compute_path_length(ee_xyz)
                path_lengths.append(path_length)

                normalised_path_length = path_length / dist_init_ee_to_goal
                normalised_path_lengths.append(normalised_path_length)

                success_ids.append(i)
                break

            inputs = update_inputs(inputs, link_pose, obs["robot_to_model"])
            decoded_output = model.decode(feature, inputs)

            if not args.skip_explicit_collision:
                if torch.sigmoid(decoded_output.collision) < args.collision_threshold:
                    diff = current_joint - prev_joint
                    out_of_bounds = np.all(diff > args.max_joint_diff) or np.all(
                        diff < -args.max_joint_diff
                    )
                    if not prev_valid_joint or out_of_bounds:
                        t0 = time.process_time()
                        has_collision, ratio = collision_between_joint_states(
                            prev_joint,
                            current_joint,
                            args.collision_threshold,
                            pc,
                            obs,
                            model,
                            scene_features,
                        )
                        t1 = time.process_time()
                        if has_collision:
                            if trials < 5:
                                # print('collision_loss: ', collision_loss)
                                # print('dist_loss: {}  ori_loss: {} coll_loss: {}'.format(geco.geco_lambda1  * dist_loss, geco.geco_lambda2 * ori_loss, geco.geco_lambda3 * collision_loss))
                                feature = prev_latent
                                opt_state_dict = opt.state_dict()
                                opt = Adam([feature], lr=args.latent_lr)
                                opt.load_state_dict(opt_state_dict)
                                geco.load_state_dict(
                                    copy.deepcopy(prev_geco_state_dict)
                                )
                                geco.geco_lambda1 *= ratio
                                decoded_output = model.decode(feature, prev_inputs)
                                trials += 1
                                colls = 0
                                continue
                            else:
                                break
                        else:
                            inter_joints = interpolate_path(
                                prev_joint, current_joint, colls
                            )
                            colls = 0
                            for joint in inter_joints:
                                inter_link_pose = robot.link_fk(
                                    np.concatenate([joint, np.array([0.04])])
                                )
                                inter_ee_pose = inter_link_pose[
                                    robot._link_map["panda_link8"]
                                ]
                                inter_ee_pos = inter_ee_pose[:3, 3]
                                trajectories["%03d" % i]["ee_jpos"].append(
                                    (joint / np.pi * 180.0).tolist()
                                )
                                trajectories["%03d" % i]["pred_ee_xyz"].append(
                                    inter_ee_pos.tolist()
                                )
                                per_frame_info = format_log(
                                    np.array(-1),
                                    1.0,
                                    np.array(-1),
                                    np.array(-1),
                                    geco.geco_lambda1,
                                    np.array(-1),
                                    geco.geco_lambda2,
                                    np.array(-1),
                                    np.array(-1),
                                    -1,
                                    -1,
                                )
                                trajectories["%03d" % i]["per_frame_info"].append(
                                    per_frame_info
                                )
                        trials = 0

                    ee_xyz.append(ee_pos)
                    per_frame_info = format_log(
                        loss_prior,
                        1.0,
                        dist_loss,
                        ori_loss,
                        geco.geco_lambda1,
                        collision_loss,
                        geco.geco_lambda2,
                        torch.sigmoid(decoded_output.collision),
                        loss,
                        dist,
                        angle,
                    )
                    trajectories["%03d" % i]["per_frame_info"].append(per_frame_info)
                    trajectories["%03d" % i]["ee_jpos"].append(
                        (current_joint / np.pi * 180.0).tolist()
                    )
                    trajectories["%03d" % i]["pred_ee_xyz"].append(ee_pos.tolist())

                else:
                    colls += 1
            else:
                per_frame_info = format_log(
                    loss_prior,
                    1.0,
                    dist_loss,
                    ori_loss,
                    geco.geco_lambda1,
                    collision_loss,
                    geco.geco_lambda2,
                    torch.sigmoid(decoded_output.collision),
                    loss,
                    dist,
                    angle,
                )
                trajectories["%03d" % i]["per_frame_info"].append(per_frame_info)
                ee_xyz.append(ee_pos)
                trajectories["%03d" % i]["ee_jpos"].append(
                    (current_joint / np.pi * 180.0).tolist()
                )
                trajectories["%03d" % i]["pred_ee_xyz"].append(ee_pos.tolist())

        i += 1

        # print("Updating output...")
        output["trajectories"] = trajectories
        output["planning_times"] = planning_times
        output["path_lengths"] = path_lengths
        output["normalised_path_lengths"] = normalised_path_lengths
        output["success_ids"] = success_ids
        output["i"] = i

        if args.save_every_traj:
            with open(save_json_path, "w") as fp:
                json.dump(output, fp, indent=2, sort_keys=True)
            print(">>> Saved output json: %s" % save_json_path)

    # prepare evaluation metrics: planning time, path length, normalised path length, success rate
    mean_time = np.mean(planning_times)
    std_time = np.std(planning_times)
    mean_path_length = np.mean(path_lengths)
    std_path_length = np.std(path_lengths)
    mean_normalised_path_length = np.mean(normalised_path_lengths)
    std_normalised_path_length = np.std(normalised_path_lengths)
    success_rate = len(success_ids) / float(num_traj)

    result = args.traj_suffix + "\n"
    result += f"Planning time: {mean_time:.4f} +- {std_time:.4f}\n"
    result += f"Path length: {mean_path_length:.4f} +- {std_path_length:.4f}\n"
    result += f"Normalised path length: {mean_normalised_path_length:.4f} +- {std_normalised_path_length:.4f}\n"
    result += f"Success rate: {success_rate * 100:.1f}\n"
    result += "Planning time: " + ", ".join(str(e) for e in planning_times) + "\n"
    result += "Path length: " + ", ".join(str(e) for e in path_lengths) + "\n"
    result += (
        "Normalised path length: "
        + ", ".join(str(e) for e in normalised_path_lengths)
        + "\n"
    )
    result += "Success id: " + ", ".join(str(e) for e in success_ids) + "\n\n"

    print(result)

    if args.dont_save:
        # save planned paths in a json file
        with open(save_json_path, "w") as fp:
            json.dump(output, fp, indent=2, sort_keys=True)
        print(">>> Saved output json: %s" % save_json_path)

        # save evaluation metrics in a txt file
        output_txt = filename + "_eval.txt"
        save_text_path = os.path.join(args.checkpt_path, output_txt)
        with open(save_text_path, "a") as f:
            f.write(result)
            f.close()

    return len(success_ids) / num_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_saved_pc", action="store_true")
    parser.add_argument("--save_every_traj", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--rollout_len", type=int, default=150)
    parser.add_argument("--traj_suffix", type=str, default="")
    parser.add_argument("--latent_lr", type=float, default=6e-2)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--rot_threshold", type=float, default=15)
    parser.add_argument("--scene_config_path", type=str, help="init config json file.")
    parser.add_argument("--collision_threshold", type=float, default=0.4)
    parser.add_argument("--skip_explicit_collision", action="store_true")
    parser.add_argument("--num_traj", type=int, default=None)
    parser.add_argument("--max_joint_diff", type=float, default=0.05)
    parser.add_argument("--dont_save", action="store_false")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

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
    evaluate(args, model, conf_module.geco_params, scene_config)
