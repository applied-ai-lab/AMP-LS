import os, sys
import copy
from collections import deque
import contextlib

import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
import trimesh
from knn_cuda import KNN
import torch_scatter
import rospy
import pcl

from core.planner.components.agent import BaseAgent
from core.utils.general_utils import ParamDict, split_along_axis, AttrDict
from core.utils.pytorch_utils import (
    map2torch,
    map2np,
    no_batchnorm_update,
    ar2ten,
    ten2ar,
)
from core.utils.vis_utils import fig2img, videos_to_grid, add_captions_to_seq
from core.data.scene_collision.src.robot import Robot
from core.utils.pytorch_transform_utils import matrix_to_quaternion
import trimesh.transformations as tra
from core.modules.geco_modules import GECO, GECO2, GECO3, GECO1
from core.utils.transform_utils import quat2mat
from core.robots.clients.panda_client import PandaClient


class LSMPAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self._update_model_params()

        self._model = self._hp.model(self._hp.model_params, logger=None)
        self.load_model_weights(
            self._model, self._hp.model_checkpoint, self._hp.model_epoch
        )

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "model": None,
                "model_params": None,
                "model_checkpoint": None,
                "model_epoch": "latest",
                "latent_lr": 3e-2,
                "online_planning": False,
                "dt": 0.1,
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _update_model_params(self):
        self._hp.model_params.device = self._hp.device
        self._hp.model_params.batch_size = 1

    def _act(self, obs):
        raise NotImplementedError

    def _act_rand(self, obs):
        return self._act(obs)

    def _compute_dist_loss(self, decoded_output, target):
        return torch.dist(decoded_output, target.detach())


class PandaLSMPAgent(LSMPAgent):
    def __init__(self, config):
        super().__init__(config)
        self.robot = Robot(
            os.path.join(self._hp.asset_path, "./data/panda/panda.urdf"),
            "panda_link8",
            device=self.device,
        )
        self.z = None
        self.feature = None
        self._opt = None
        self.geco = GECO2(self._hp.geco_params)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "asset_path": "./core/data/scene_collision/assets/",
                "has_rot": False,
                "debug": False,
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        self.robot.set_joint_cfg(obs["robot_states"])
        ee_pose = self.robot.ee_pose
        ee_pos = ee_pose[:, :3, 3]
        repr_6d = ee_pose[:, :3, :2].permute((0, 2, 1)).reshape((-1, 6))
        if self._hp.has_rot:
            states = np.concatenate(
                [obs["robot_states"][:7][None], ten2ar(ee_pos), ten2ar(repr_6d)],
                axis=-1,
            )
        else:
            states = np.concatenate(
                [obs["robot_states"][:7][None], ten2ar(ee_pos)], axis=-1
            )

        inputs = AttrDict(
            states=states.reshape((1, 1, states.shape[-1])),
        )
        inputs = map2torch(inputs, device=self.device)

        # if self.z is None or self._hp.online_planning:
        if self.z is None:
            with torch.no_grad():
                encoded_output = self._model.encode(inputs)
                self.z = encoded_output.q.mu.detach()
            self.feature = nn.Parameter(self.z)
            self._opt = Adam([self.feature], lr=self._hp.latent_lr)

        decoded_output = self._model.decode(self.feature)
        current_joint = decoded_output.squeeze(0)[:, :7]
        current_ee = decoded_output.squeeze(0)[:, 7:10]
        current_ori = decoded_output.squeeze(0)[:, 10:]

        self._opt.zero_grad()
        dist_loss = self._compute_dist_loss(
            current_ee, ar2ten(obs["goal"].astype(np.float32), self.device).unsqueeze(0)
        )
        ori_loss = nn.MSELoss()(
            current_ori,
            ar2ten(
                obs["goal_rot"][:3, :2].transpose().reshape(-1).astype(np.float32),
                self.device,
            )
            .unsqueeze(0)
            .detach(),
        )
        loss_prior = F.mse_loss(self.feature, torch.zeros_like(self.feature))
        loss = self.geco.loss(loss_prior, dist_loss, ori_loss)
        if self._hp.debug:
            print(ee_pos, " goal:  ", obs["goal"])
            print("Loss: ", loss)
        loss.backward()
        self._opt.step()

        next_joint = ten2ar(self._model.decode(self.feature).squeeze(0)[0])[:7]
        policy_output = AttrDict(action=next_joint)
        return policy_output

    def reset(self):
        self._z = None
        self.feature = None
        self._opt = None


class PandaSceneLSMPAgent(PandaLSMPAgent):
    def __init__(self, config):
        super().__init__(config)
        self.robot = Robot(
            os.path.join(self._hp.asset_path, "./data/panda/panda.urdf"),
            "panda_link8",
            device=self.device,
        )
        self.z = None
        self.feature = None
        self._opt = None
        self.cur_scene_pc = None
        self.knn = None
        self.geco = GECO2(self._hp.geco_params)
        self.geco.to(self.device)
        self.coll_counts = 0
        self.client = PandaClient(AttrDict(set_planning_time=0.01))
        self.prev_latent = None
        self.inter_joints = None
        self.next_decoded_output = None
        self.knn = KNN(k=8, transpose_mode=True).to(self.device)

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "geco_params": AttrDict(),
                "aggregate_pc": True,
                "collision_threshold": 0.3,
                "ori_scale": 1.0,
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _aggregate_pc(self, cur_pc, new_pc):
        # Filter tiny clusters of points and split into vis/occluding
        cam_model_tr = torch.from_numpy(self.camera_pose[:3, 3]).float().to(self.device)
        new_pc = torch.from_numpy(new_pc).float().to(self.device)
        if self.knn is not None:
            nearest = self.knn(new_pc[None, ...], new_pc[None, ...])[0][0, :, -1]

        vis_mask = torch.from_numpy(self.scene_pc_mask).to(self.device)
        dists = torch.norm(new_pc - cam_model_tr, dim=1) ** 2
        dists /= dists.max()
        if self.knn is None:
            nearest = torch.zeros_like(vis_mask)
        occ_scene_pc = new_pc[~vis_mask & (nearest < 0.1 * dists)]
        scene_pc = new_pc[vis_mask & (nearest < 0.1 * dists)]

        if cur_pc is not None:
            # Group points by rays; get mapping from points to unique rays
            cur_pc_rays = cur_pc - cam_model_tr
            cur_pc_dists = torch.norm(cur_pc_rays, dim=1, keepdim=True) + 1e-12
            cur_pc_rays /= cur_pc_dists
            occ_pc_rays = occ_scene_pc - cam_model_tr
            occ_pc_dists = torch.norm(occ_pc_rays, dim=1, keepdim=True) + 1e-12
            occ_pc_rays /= occ_pc_dists
            occ_rays = (
                (torch.cat((cur_pc_rays, occ_pc_rays), dim=0) * 50).round().long()
            )
            _, occ_uniq_inv, occ_uniq_counts = torch.unique(
                occ_rays, dim=0, return_inverse=True, return_counts=True
            )

            # Build new point cloud from previous now-occluded points and new pc
            cur_occ_inv = occ_uniq_inv[: len(cur_pc_rays)]
            cur_occ_counts = torch.bincount(cur_occ_inv, minlength=len(occ_uniq_counts))
            mean_occ_dists = torch_scatter.scatter_max(
                occ_pc_dists.squeeze(),
                occ_uniq_inv[-len(occ_pc_rays) :],
                dim_size=occ_uniq_inv.max() + 1,
            )[0]
            occ_mask = (occ_uniq_counts > cur_occ_counts) & (cur_occ_counts > 0)
            occ_pc = cur_pc[
                occ_mask[cur_occ_inv]
                & (cur_pc_dists.squeeze() > mean_occ_dists[cur_occ_inv] + 0.01)
            ]
            return torch.cat((occ_pc, scene_pc), dim=0)
        else:
            return scene_pc

    def _prepare_pc(self, obs):
        label_map = obs["label_map"]
        scene_labels = obs["pc_label"]
        self.scene_pc_mask = np.logical_and(
            scene_labels != label_map["robot"], scene_labels != label_map["target"]
        )
        self.camera_pose = obs["camera_pose"]
        scene_pc = tra.transform_points(obs["pc"], self.camera_pose)
        # scene_pc = obs['pc']

        if self.cur_scene_pc is not None and self._hp.aggregate_pc:
            self.cur_scene_pc = self._aggregate_pc(self.cur_scene_pc, scene_pc)
        else:
            self.cur_scene_pc = self._aggregate_pc(None, scene_pc)

        model_scene_pc = (
            ar2ten(obs["robot_to_model"].astype(np.float32), self.device)
            @ torch.cat(
                [
                    self.cur_scene_pc,
                    torch.ones((len(self.cur_scene_pc), 1)).to(self.device),
                ],
                axis=1,
            ).T
        )
        model_scene_pc = model_scene_pc[:3].T

        # model_scene_pc = ar2ten(model_scene_pc.astype(np.float32), self._hp.device)
        # Clip points to model bounds and feed in for features
        in_bounds = (model_scene_pc[..., :3] > self._model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (model_scene_pc[..., :3] < self._model.bounds[1] - 1e-5).all(
            dim=-1
        )
        return model_scene_pc[in_bounds]

    def _act(self, obs):
        if self.inter_joints is not None:
            action = self.inter_joints[self.inter_idx]
            self.inter_idx += 1
            if self.inter_idx >= len(self.inter_joints):
                self.inter_joints = None
        else:
            if self._hp.online_planning or (
                not self._hp.online_planning and self.scene_pc is None
            ):
                scene_pc = self._prepare_pc(obs)
                self.scene_pc = scene_pc

            if self.next_joint is None:
                self.next_joint = obs["robot_states"][:7]
            self.robot.set_joint_cfg(
                ar2ten(
                    np.concatenate([self.next_joint, np.array([0.04])]).astype(
                        np.float32
                    ),
                    self.device,
                )
            )
            ee_pose = self.robot.ee_pose
            ee_pos = ee_pose[:, :3, 3]
            repr_6d = ee_pose[:, :3, :2].permute((0, 2, 1)).reshape((-1, 6))

            states = torch.cat(
                [ar2ten(self.next_joint[None], self.device), ee_pos, repr_6d], dim=-1
            )
            inputs = AttrDict(
                states=states.reshape((1, 1, states.shape[-1])),
                scene_pc=self.scene_pc.reshape((1, 1, *self.scene_pc.shape)),
            )

            inputs = self._update_inputs(obs, inputs)
            inputs = map2torch(inputs, device=self.device)

            if self.z is None:
                with torch.no_grad():
                    encoded_output = self._model.encode(inputs)
                    self.z = encoded_output.q.mu.detach()
                self.feature = nn.Parameter(self.z)
                self._opt = Adam([self.feature], lr=self._hp.latent_lr)
                self.prev_latent = nn.Parameter(self.feature.clone().detach())
                self.prev_valid_joint = True
                self.prev_joint = self.next_joint
                self.prev_geco_state_dict = self.geco.state_dict()

            decoded_output = self._model.decode(self.feature, inputs)
            current_joint = decoded_output.states.squeeze(0)[:, :7]
            current_ee = decoded_output.states.squeeze(0)[:, 7:10]
            current_ori = decoded_output.states.squeeze(0)[:, 10:]
            self._opt.zero_grad()

            dist_loss = self._compute_dist_loss(
                current_ee,
                ar2ten(obs["goal"].astype(np.float32), self.device).unsqueeze(0),
            )
            ori_loss = nn.MSELoss()(
                current_ori,
                ar2ten(
                    obs["goal_rot"][:3, :2].transpose().reshape(-1).astype(np.float32),
                    self.device,
                )
                .unsqueeze(0)
                .detach(),
            )
            loss_prior = F.mse_loss(self.feature, torch.zeros_like(self.feature))
            collision_loss = torch.binary_cross_entropy_with_logits(
                decoded_output.collision, torch.zeros_like(decoded_output.collision)
            ).sum()

            target_loss = dist_loss + self._hp.ori_scale * ori_loss
            loss = self.geco.loss(loss_prior, target_loss, collision_loss)
            loss.backward()
            self._opt.step()

            next_joint = self._model.decode_states(self.feature).squeeze(0)[0][:7]
            self.robot.set_joint_cfg(
                torch.cat(
                    [
                        next_joint,
                        ar2ten(np.array([0.04]).astype(np.float32), self.device),
                    ]
                )
            )
            inputs = self._update_inputs(obs, inputs)
            next_decoded_output = self._model.decode(self.feature, inputs)
            self.next_decoded_output = next_decoded_output

            print(
                "Next Collision probability: ",
                ten2ar(torch.sigmoid(next_decoded_output.collision)),
            )
            self.next_joint = ten2ar(next_joint)
            if (
                torch.sigmoid(next_decoded_output.collision)
                < self._hp.collision_threshold
            ):
                if not self.prev_valid_joint:
                    has_collision, ratio = self.collision_between_joint_states(
                        self.prev_joint, self.next_joint, obs
                    )
                    if has_collision:
                        del self.feature
                        self.feature = self.prev_latent
                        opt_state_dict = self._opt.state_dict()
                        self._opt = Adam([self.feature], lr=self._hp.latent_lr)
                        self._opt.load_state_dict(opt_state_dict)
                        self.geco.load_state_dict(
                            copy.deepcopy(self.prev_geco_state_dict)
                        )
                        self.geco.geco_lambda1 *= ratio
                        self.next_joint = self.prev_joint
                        action = np.zeros_like(obs["robot_states"][:7])
                    else:
                        self.inter_joints = self.interpolate_path(
                            self.prev_joint, self.next_joint, self.coll_counts
                        )
                        self.inter_idx = 0
                        action = self.inter_joints[self.inter_idx]
                        self.inter_idx += 1
                        if self.inter_idx >= len(self.inter_joints):
                            self.inter_joints = None
                else:
                    action = next_joint

                self.prev_valid_joint = True
                self.prev_joint = copy.deepcopy(self.next_joint)
                self.prev_latent = nn.Parameter(self.feature.clone().detach())
                self.prev_geco_state_dict = self.geco.state_dict()
            else:
                self.coll_counts += 1
                action = np.zeros_like(obs["robot_states"][:7])
                self.prev_valid_joint = False

        policy_output = AttrDict(action=action)
        return policy_output

    def _update_inputs(self, obs, inputs):
        trans = []
        rots = []
        # for i, link in enumerate(self.robot.mesh_links[1:]):
        for i, link in enumerate(reversed(self.robot.mesh_links[1:])):
            poses_tf = (
                ar2ten(obs["robot_to_model"].astype(np.float32), self.device)
                @ self.robot.link_poses[link]
            )
            trans.append(poses_tf[:, :3, 3])
            rots.append(poses_tf[:, :3, :2].reshape(1, -1))
        trans = torch.cat(trans).unsqueeze(0)
        rots = torch.cat(rots).unsqueeze(0)

        inputs.update(AttrDict(trans=trans.unsqueeze(0), rots=rots.unsqueeze(0)))
        return inputs

    def reset(self):
        self.coll_counts = 0
        self.valid_move = True
        self.next_joint = None
        self.scene_pc = None

    def interpolate_path(self, prev_joint, current_joint, timesteps):
        diff = (current_joint - prev_joint) / (timesteps + 1)
        intervals = diff * np.arange(1, timesteps + 2)[:, None]
        inter_joints = prev_joint[None].repeat(len(intervals), 0) + intervals
        return inter_joints

    def collision_between_joint_states(
        self, prev_joint, current_joint, obs, samples=15
    ):
        current_joint = ten2ar(current_joint)
        prev_joint = ten2ar(prev_joint)

        diff = (current_joint - prev_joint) / (samples + 1)
        joint_states = (
            prev_joint[None].repeat(samples, 0)
            + diff[None].repeat(samples, 0) * np.arange(1, samples + 1)[:, None]
        )
        self.robot.set_joint_cfg(
            ar2ten(
                np.concatenate(
                    [
                        joint_states.astype(np.float32),
                        np.ones((samples, 1)).astype(np.float32) * 0.04,
                    ],
                    axis=1,
                ),
                self.device,
            )
        )

        ee_pose = self.robot.ee_pose
        ee_pos = ee_pose[:, :3, 3]
        ee_6d = ee_pose[:, :3, :2].permute((0, 2, 1)).reshape(len(ee_pose), -1)

        inputs = AttrDict(
            states=torch.cat(
                [ar2ten(joint_states.astype(np.float32), self.device), ee_pos, ee_6d],
                dim=-1,
            ).unsqueeze(0),
            scene_pc=self.scene_pc.reshape((1, 1, *self.scene_pc.shape)),
        )
        trans = []
        rots = []
        robot_to_model = ar2ten(obs["robot_to_model"].astype(np.float32), self.device)
        # for i, link in enumerate(self.robot.mesh_links[1:]):
        for i, link in enumerate(reversed(self.robot.mesh_links[1:])):
            pose = robot_to_model @ self.robot.link_poses[link]
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
        with torch.no_grad():
            encoded_output = self._model.encode(inputs)
            z = encoded_output.q.mu.detach()
            output = self._model.decode(z, inputs)
        collision_prob = torch.sigmoid(output.collision)
        if torch.sum(collision_prob > self._hp.collision_threshold) > 0:
            indices = torch.nonzero(collision_prob > self._hp.collision_threshold)
            return True, (ten2ar(indices[0][1] + 1)) / samples
        else:
            return False, 1
