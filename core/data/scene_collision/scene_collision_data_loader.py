import os, sys
import time
import glob
import itertools
import numpy as np
import random
import tqdm
import h5py
import cv2
import trimesh.transformations as tra
from autolab_core import CameraIntrinsics
from torch.utils.data import IterableDataset
from trimesh.collision import CollisionManager
from urdfpy import URDF

from core.utils.general_utils import AttrDict
from core.utils.pytorch_utils import ar2ten, ten2ar
from core.components.data_loader import (
    GlobalSplitVideoDataset,
    OfflineDataset,
    OfflineVideoDataset,
)
from core.data.scene_collision.src.scene import SceneManager, SceneRenderer
from core.data.scene_collision.src.robot import Robot
from core.data.scene_collision.utils.scene_collision_utils import compute_camera_pose
from core.utils.general_utils import ForkedPdb
from core.utils.transform_utils import mat2quat, mat2euler


class SceneCollisionDataset(GlobalSplitVideoDataset):
    DATA_KEYS = ["pad_mask", "joints", "ee_pos", "ee_rot", "ee_quat"]

    def _load_raw_data(self, data, F):
        assert (
            self.samples_per_file == 1
        )  # assume that all files have just one trajectory
        key = "traj0"

        # Fetch data into a dict
        for name in F[key].keys():
            if name in self.DATA_KEYS:
                # data[name] = F[key + '/' + name][()].astype(np.float32)
                raw_data = F[key + "/" + name][()]
                if not isinstance(raw_data, str):
                    raw_data = raw_data.astype(np.float32)
                data[name] = raw_data
        repr_6d = data.ee_rot[:, :3, :2].transpose((0, 2, 1)).reshape(-1, 6)
        data.states = np.concatenate([data.joints, data.ee_pos, repr_6d], axis=-1)
        data.images = np.zeros((len(raw_data), 32, 32, 3)).astype(np.uint8)


class SceneCollisionDataset(OfflineVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = AttrDict()
        self.meshes = self.spec.meshes
        self.query_size = self.spec.query_size
        self.cam_intr = CameraIntrinsics(**self.spec.intrinsics)
        self.cam_pose = self.spec.extrinsics
        self.bounds = np.array(self.spec.bounds)
        self.batch_size = 1

        self.n_scene_points = self.spec.n_scene_points

        self.robot = URDF.load(self.spec.robot_urdf)
        # Get link poses for a series of random configurations
        low_joint_limits, high_joint_limits = self.robot.joint_limit_cfgs
        self.low_joint_vals = np.fromiter(low_joint_limits.values(), dtype=float)
        self.high_joint_vals = np.fromiter(high_joint_limits.values(), dtype=float)
        self.low_joint_vals[-1] = 0.04

        meshes = self.robot.collision_trimesh_fk().keys()
        self.link_meshes = list(meshes)
        self.num_links = len(self.link_meshes)
        self.link_combos = list(itertools.combinations(range(len(meshes)), 2))
        self.links = [
            link for link in self.robot.links if link.collision_mesh is not None
        ]

        # Add the meshes to the collision managers
        self.self_collision_managers = []
        for m in self.link_meshes:
            collision_manager = CollisionManager()
            collision_manager.add_object("link", m)
            self.self_collision_managers.append(collision_manager)

    # Generator that yields batches of training tuples
    def get_sample(self):
        """
        Generator that yields batches of training tuples
        Outputs:
            scene_points (self.batch_size, self.n_scene_points, 3): scene point cloud batch
            obj_points (self.n_obj_points, 3): object point cloud batch
            trans (self.query_size, 3): translations of object in scene
            rots (self.query_size, 6): rotations of object in scene (first two cols of rotation matrix)
            colls (self.query_size,): boolean array of GT collisions along trajectories
            scene_manager (utils.SceneManager): underlying scene manager for GT collisions
        """

        camera_pose = self.sample_camera_pose(mean=self.spec.camera_pose_mean)
        scene_points, scene_manager = self.get_scene(
            low=self.spec.n_obj_low, high=self.spec.n_obj_high, camera_pose=camera_pose
        )
        np.random.seed()
        colls, states, trans, rots = self.get_colls(
            scene_manager,
        )
        del scene_manager

        data_dict = AttrDict(
            states=states.astype(np.float32),
            scene_pc=scene_points.astype(np.float32),
            collision=colls,
            trans=trans.astype(np.float32),
            rots=rots.astype(np.float32),
        )
        return data_dict

    def get_scene(self, low=2, high=8, camera_pose=None):
        """
        Generate a scene point cloud by placing meshes on a tabletop
        Inputs:
            low (int): minimum number of objects to place
            high (int): maximum number of objects to place
            camera_pose (4, 4): optional camera pose
        Outputs:
            points_batch (self.batch_size, self.n_scene_points, 3): scene point cloud batch
            scene_manager (utils.SceneManager): underlying scene manager for GT collisions
        """
        # Create scene with random number of objects btw low and high
        num_objs = np.random.randint(low, high)

        scene_manager = self._create_scene()
        robot_to_model = np.eye(4)
        robot_to_model[:3, 3] = (-0.6, 0, scene_manager._table_dims[2] / 2.0)
        pose = np.eye(4)
        scene_manager._collision_manager.add_object(
            "query_object", self.links[0].collision_mesh, robot_to_model @ pose
        )
        scene_manager.arrange_scene(int(num_objs))
        scene_manager._collision_manager.remove_object("query_object")

        # Render points from batch_size different angles
        points_batch = np.zeros(
            (self.batch_size, self.n_scene_points, 3), dtype=np.float32
        )
        for i in range(self.batch_size):
            if camera_pose is None:
                camera_pose = self.sample_camera_pose()
            scene_manager.camera_pose = camera_pose

            points = scene_manager.render_points()
            points = points[(points[:, :3] > self.bounds[0] + 1e-4).all(axis=1)]
            points = points[(points[:, :3] < self.bounds[1] - 1e-4).all(axis=1)]
            pt_inds = np.random.choice(points.shape[0], size=self.n_scene_points)
            sample_points = points[pt_inds]
            points_batch[i] = sample_points

        return points_batch, scene_manager

    def get_colls(
        self,
        scene_manager,
    ):
        """
        Generate object/scene collision trajectories
        Inputs:
            scene_manager (utils.SceneManager): Underlying scene manager
            obj (trimesh.Trimesh): Underlying object mesh
            obj_pose (4, 4): Object GT pose matrix
            obj_centroid (3,): Centroid of object point cloud
        Outputs:
            trans (self.query_size, 3): translations of object in scene
            rots (self.query_size, 6): rotations of object in scene (first two cols of rotation matrix)
            colls (self.query_size,): boolean array of GT collisions along trajectories
        """

        # st = time.time()
        # Apply to object within scene and find GT collision status
        robot_to_model = np.eye(4)
        robot_to_model[:3, 3] = (-0.6, 0, scene_manager._table_dims[2] / 2.0)
        colls = []
        states = []
        rand_cfgs = (
            np.random.rand(self.query_size * 4, len(self.low_joint_vals))
            * (self.high_joint_vals - self.low_joint_vals)
            + self.low_joint_vals
        )
        ee_poses = robot_to_model @ self.robot.link_fk_batch(
            rand_cfgs, link="panda_link8"
        )
        ee_pos = ee_poses[:, :3, 3]
        in_bounds = (ee_pos > self.bounds[0] + 1e-5).all(axis=1)
        in_bounds &= (ee_pos < self.bounds[1] - 1e-5).all(axis=1)
        rand_cfgs = rand_cfgs[in_bounds]

        new_rand_cfgs = (
            np.random.rand(self.query_size, len(self.low_joint_vals))
            * (self.high_joint_vals - self.low_joint_vals)
            + self.low_joint_vals
        )
        rand_cfgs = np.concatenate([rand_cfgs, new_rand_cfgs])

        link_poses = self.robot.link_fk_batch(rand_cfgs)
        indices = np.random.permutation(np.arange(len(rand_cfgs)))
        non_self_coll_indices = []

        valid_joints = []
        colls = np.zeros((self.query_size, len(self.links[1:]))).astype(bool)
        trans = []
        rots = []
        for k in indices:
            self_coll = self.check_pairwise_distances(link_poses, k)
            if not self_coll:
                valid_joints.append(rand_cfgs[k])
                ee_pose = link_poses[self.robot._link_map["panda_link8"]][k]
                pos = ee_pose[:3, 3]
                if self.spec.has_rot:
                    repr_6d = ee_pose[:3, :2].transpose().reshape(-1)
                    states.append(np.concatenate([rand_cfgs[k, :-1], pos, repr_6d]))
                else:
                    states.append(np.concatenate([rand_cfgs[k, :-1], pos]))
            if len(valid_joints) == self.query_size:
                break

        # et = time.time()
        # elapsed_time = et - st
        # print('Execution time before coll check:', elapsed_time, 'seconds')
        valid_joints = np.array(valid_joints)

        link_poses = self.robot.link_fk_batch(valid_joints)
        for i, link in enumerate(reversed(self.links[1:])):
            # scene_manager._collision_manager.add_object('query_object', link.collision_mesh)
            for k, joint in enumerate(valid_joints):
                if np.sum(colls[k][:i]) == 0:
                    pose = link_poses[link][k]
                    # scene_manager._collision_manager.set_transform('query_object', robot_to_model@pose)
                    self.self_collision_managers[len(self.links) - i - 1].set_transform(
                        "link", robot_to_model @ pose
                    )  # skip link_0
                    # coll = scene_manager.collides()
                    coll = scene_manager._collision_manager.in_collision_other(
                        self.self_collision_managers[len(self.links) - i - 1]
                    )
                    colls[k][i] = coll
            # scene_manager._collision_manager.remove_object('query_object')
            pose = robot_to_model @ link_poses[link]
            trans.append(pose[:, :3, 3])
            rots.append(pose[:, :3, :2].reshape(len(pose), -1))

        colls = np.sum(colls, axis=1).astype(bool)

        trans = np.array(trans).transpose((1, 0, 2))
        rots = np.array(rots).transpose((1, 0, 2))
        # et = time.time()
        # elapsed_time = et - st
        # print('Execution time:', elapsed_time, 'seconds')
        return np.array(colls), np.array(states), trans, rots

    # Creates scene manager and renderer
    def _create_scene(self):
        r = SceneRenderer()
        r.create_camera(self.cam_intr, znear=0.04, zfar=5)
        s = SceneManager(self.meshes, renderer=r)
        return s

    # Samples a camera pose that looks at the center of the scene
    def sample_camera_pose(self, mean=False):
        if mean:
            az = np.mean(self.cam_pose["azimuth"])
            elev = np.mean(self.cam_pose["elevation"])
            radius = np.mean(self.cam_pose["radius"])
        else:
            az = np.random.uniform(*self.cam_pose["azimuth"])
            elev = np.random.uniform(*self.cam_pose["elevation"])
            radius = np.random.uniform(*self.cam_pose["radius"])

        sample_pose, _ = compute_camera_pose(radius, az, elev)

        return sample_pose

    def check_pairwise_distances(self, mesh_poses, ind):
        coll = False
        for _, (i, j) in enumerate(self.link_combos):
            if abs(i - j) < 2 or ((i, j) == (6, 8)) or ((i, j) == (8, 10)):
                continue
            i_tf = mesh_poses[self.links[i]][ind]
            self.self_collision_managers[i].set_transform("link", i_tf)
            j_tf = mesh_poses[self.links[j]][ind]
            self.self_collision_managers[j].set_transform("link", j_tf)
            coll |= self.self_collision_managers[i].in_collision_other(
                self.self_collision_managers[j]
            )
            if coll:
                return coll
        return coll

    def __len__(self):
        if self.phase == "train":
            return 10000
        else:
            return 40


class RobotCollisionFreeJointDataset(OfflineVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = AttrDict()
        self.robot = URDF.load(self.spec.robot_urdf)

        # Get link poses for a series of random configurations
        low_joint_limits, high_joint_limits = self.robot.joint_limit_cfgs
        self.low_joint_vals = np.fromiter(low_joint_limits.values(), dtype=float)
        self.high_joint_vals = np.fromiter(high_joint_limits.values(), dtype=float)

        self.low_joint_vals[-1] = 0.04
        meshes = self.robot.collision_trimesh_fk().keys()
        self.link_meshes = list(meshes)
        self.num_links = len(self.link_meshes)
        self.link_combos = list(itertools.combinations(range(len(meshes)), 2))

        # Add the meshes to the collision managers
        self.collision_managers = []
        for m in self.link_meshes:
            collision_manager = CollisionManager()
            collision_manager.add_object("link", m)
            self.collision_managers.append(collision_manager)

    # Generator that yields batches of training tuples
    def get_sample(self):
        data_dict = AttrDict()
        coll = True
        while coll:
            rand_cfg = (
                np.random.rand(len(self.low_joint_vals))
                * (self.high_joint_vals - self.low_joint_vals)
                + self.low_joint_vals
            )
            mesh_pose = self.robot.collision_trimesh_fk(rand_cfg)
            ee_pose = self.robot.link_fk(rand_cfg, link="panda_link8")
            pos = ee_pose[:3, 3]
            R = ee_pose[:3, :3]
            repr6d = R[:, :2].transpose().reshape(-1)
            coll = self.check_pairwise_distances(mesh_pose, boolean=True)

        # if self.spec.has_rot:
        #     data_dict['states'] = np.concatenate([rand_cfg[:-1], pos, repr6d])[None].astype(np.float32)
        # else:
        #     data_dict['states'] = np.concatenate([rand_cfg[:-1], pos])[None].astype(np.float32)
        states = [rand_cfg[:-1]]
        if self.spec.has_pos:
            states.append(pos)
        if self.spec.has_rot:
            states.append(repr6d)
        data_dict["states"] = np.concatenate(states)[None].astype(np.float32)
        data_dict["pos"] = pos[None].astype(np.float32)
        data_dict["rot"] = repr6d[None].astype(np.float32)
        return data_dict

    def check_pairwise_distances(self, mesh_poses, boolean=False, normalize=False):
        coll = False
        for _, (i, j) in enumerate(self.link_combos):
            if abs(i - j) < 2 or ((i, j) == (6, 8)) or ((i, j) == (8, 10)):
                continue
            i_tf = mesh_poses[self.link_meshes[i]]
            self.collision_managers[i].set_transform("link", i_tf)
            j_tf = mesh_poses[self.link_meshes[j]]
            self.collision_managers[j].set_transform("link", j_tf)
            coll |= self.collision_managers[i].in_collision_other(
                self.collision_managers[j]
            )
            if coll:
                return coll
        return coll

    def __len__(self):
        if self.phase == "train":
            return 200000
        else:
            return 2048
