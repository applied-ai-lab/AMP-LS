import os
import numpy as np
import h5py
import trimesh
import gym

import trimesh
import itertools
from urdfpy import URDF

from trimesh.collision import CollisionManager
from autolab_core import CameraIntrinsics
from core.data.scene_collision.src.scene import SceneManager, SceneRenderer
from core.data.scene_collision.utils.scene_collision_utils import compute_camera_pose
from core.utils.general_utils import AttrDict, ParamDict
from core.utils.transform_utils import mat2quat


class ScenePandaEnv(gym.Env):
    def __init__(self, config=AttrDict()) -> None:
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)
        self.cam_pose = AttrDict(
            azimuth=[-0.2, 0.2],
            elevation=[0.6, 1.0],
            radius=[1.5, 2.0],
        )

        self.scene_manager = None
        obj_info = h5py.File(
            os.path.join(self._hp.shapenet_datadir, "./object_info.hdf5"), "r"
        )
        self.mesh_info = obj_info["meshes"]
        self.categories = obj_info["categories"]

        self.robot = URDF.load(self._hp.robot_urdf)
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

    def _default_hparams(self):
        default_params = ParamDict(
            {
                "shapenet_datadir": os.path.join(
                    os.environ["DATA_DIR"], "./shapenet_x1.5"
                ),
                "robot_urdf": "./core/data/scene_collision/assets/data/panda/panda.urdf",
                "max_num_objs": 10,
                "min_num_objs": 6,
                "n_points": 30000,  #  number of points for point cloud
                "bg_seg_label": 0,
                "env_seg_label": 1,
                "robot_seg_label": 2,
                "target_seg_label": 3,
                "robot_client_params": AttrDict(),
                "robot_to_model": (-0.6, 0, 0.25),
                "bounds": np.array([[-0.5, -0.8, 0.24], [0.5, 0.8, 0.6]]),
            }
        )
        return default_params

    def reset(self):
        self.create_scene()
        robot_to_model = np.eye(4)
        robot_to_model[:3, 3] = self._hp.robot_to_model
        pose = np.eye(4)
        self.scene_manager._collision_manager.add_object(
            "query_object", self.links[0].collision_mesh, robot_to_model @ pose
        )
        n_objs = np.random.randint(4, 8)
        # self.scene_manager.arrange_scene(n_objs)
        self.scene_manager.arrange_single_obs_scene()
        self.scene_manager._collision_manager.remove_object("query_object")

        pc = self.build_pc_obs()
        robot_states = self.sample_valid_state(
            bounds=np.array([[0, -0.8, 0.24], [0.5, 0, 0.5]])
        )
        goal_states = self.sample_valid_state(
            bounds=np.array([[0, 0, 0.24], [0.5, 0.8, 0.5]])
        )

        return AttrDict(
            robot_states=np.concatenate([robot_states[:7], np.array([0.04])]),
            goal=robot_states[7:10],
            pc=pc,
            robot_to_model=robot_to_model,
            label_map=AttrDict(
                robot=self._hp.robot_seg_label,
                env=self._hp.env_seg_label,
                bg=self._hp.bg_seg_label,
                target=self._hp.target_seg_label,
            ),
            pc_label=np.ones(len(pc)),
        )

    def arrange_objects(self):
        n_objs = np.random.randint(self._hp.min_num_objs, self._hp.max_num_objs)

        robot_to_model = np.eye(4)
        pose = np.eye(4)
        self.create_scene()
        self.scene_manager._collision_manager.add_object(
            "query_object", self.links[0].collision_mesh, robot_to_model @ pose
        )
        self.scene_manager.arrange_scene(n_objs)
        self.scene_manager._collision_manager.remove_object("query_object")

        if len(self.scene_manager.objs) == 0:
            raise ValueError("Environment does not have any objects!")

        scene_info = []

        for i, (obj_name, obj_info) in enumerate(self.scene_manager.objs.items()):
            if obj_name == "table":
                continue
            else:
                obj_output = {}
                obj_output["id"] = i
                obj_output["name"] = obj_name
                obj_output["pose"] = obj_info[
                    "pose"
                ].tolist()  # ndarray is not serializable when saving as json
                obj_output["path"] = obj_info["path"]

                scene_info.append(obj_output)

        return scene_info

    def arrange_objects_to(self, scene_info):
        self.create_scene()
        self.scene_manager.arrange_table()

        for obj_input in scene_info:
            obj_id = "obj_{:d}".format(obj_input["id"])

            mesh_path = os.path.join(self._hp.shapenet_datadir, obj_input["path"])
            mesh = trimesh.load(mesh_path, force="mesh")

            pose = np.array(obj_input["pose"])  # convert list back to ndarray

            self.scene_manager.add_object(name=obj_id, mesh=mesh, pose=pose)
        # import pyrender
        # pyrender.viewer.Viewer(self.scene_manager._renderer._scene, viewer_flags={'use_direct_lighting': True, 'lighting_intensity': 1.0})

    def has_collision(self, joints):
        robot_to_model = np.eye(4)
        link_pose = self.robot.link_fk(np.concatenate([joints, np.array([0.04])]))
        coll = False
        for link in self.links[1:]:
            pose = link_pose[link]
            self.scene_manager._collision_manager.add_object(
                "query_object", link.collision_mesh, robot_to_model @ pose
            )
            # self.scene_manager._renderer.add_object('query_object', link.collision_mesh, robot_to_model@pose)
            # import pyrender
            # pyrender.viewer.Viewer(self.scene_manager._renderer._scene, viewer_flags={'use_direct_lighting': True, 'lighting_intensity': 1.0})
            coll = coll or self.scene_manager.collides()
            self.scene_manager._collision_manager.remove_object("query_object")
            if coll:
                return coll
        return coll

    def sample_valid_state(self, bounds=None):
        """
        The configuration is stored as a Numpy array with one row in the format
        [j1, j2, ..., j7, ee_x, ee_y, ee_z].
        Cases where there is self-collision or where the end-effector is under the table are discarded.
        """
        result = np.zeros(10)
        robot_to_model = np.eye(4)
        # robot_to_model[:3, 3] = self._hp.robot_to_model

        while True:
            # print('sample')
            # sample a random joint configuration
            rand_cfg = (
                np.random.rand(len(self.low_joint_vals))
                * (self.high_joint_vals - self.low_joint_vals)
                + self.low_joint_vals
            )
            ee_pose = robot_to_model @ self.robot.link_fk(rand_cfg, link="panda_link8")

            if bounds is not None:
                ee_pos = ee_pose[:3, 3]
                in_bound = (ee_pos > bounds[0] + 1e-5).all()
                in_bound &= (ee_pos < bounds[1] - 1e-5).all()
                if not in_bound:
                    continue

            link_pose = self.robot.link_fk(rand_cfg)
            self_coll = self.check_pairwise_distance(link_pose)
            if self_coll:
                continue
            coll = False
            for link in self.links[1:]:
                pose = link_pose[link]
                self.scene_manager._collision_manager.add_object(
                    "query_object", link.collision_mesh, robot_to_model @ pose
                )
                coll = coll or self.scene_manager.collides()
                self.scene_manager._collision_manager.remove_object("query_object")
            if not coll:
                break

        # import pyrender
        # pyrender.viewer.Viewer(self.scene_manager._renderer._scene, viewer_flags={'use_direct_lighting': True, 'lighting_intensity': 1.0})

        ee_pose = self.robot.link_fk(rand_cfg, link="panda_link8")
        return np.concatenate([rand_cfg[:7], ee_pose[:3, 3], mat2quat(ee_pose[:3, :3])])

    def create_scene(self):
        # scene manager
        if self.scene_manager is not None:
            del self.scene_manager
        r = SceneRenderer()
        intrinsics = AttrDict(
            frame="camera",
            fx=616.36529541,
            fy=616.20294189,
            cx=310.25881958,
            cy=236.59980774,
            skew=0.0,
            width=640,
            height=480,
        )
        cam_intr = CameraIntrinsics(**intrinsics)
        r.create_camera(cam_intr, znear=0.04, zfar=5)
        self.scene_manager = SceneManager(
            "./datasets/shapenet_x1.5",
            renderer=r,
            cat=[
                "1Shelves",
                "2Shelves",
                "3Shelves",
                "4Shelves",
                "5Shelves",
                "6Shelves",
                "7Shelves",
                "AccentChair",
                "AccentTable",
                "Bed",
                "Bench",
                "Cabinet",
                "Chair",
                "ChestOfDrawers",
                "ChildBed",
                "CoffeeTable",
                "Couch",
                "CurioCabinet",
                "Desk",
                "Desktop",
                "DiningTable",
                "DoubleBed",
                "DraftingTable",
                "Dresser",
                "DresserWithMirror",
                "EndTable",
                "GameTable",
                "Gamecube",
                "KneelingChair",
                "Laptop",
                "MediaChest",
                "Microwave",
                "Monitor",
                "OfficeChair",
                "OfficeSideChair",
                "Oven",
                "OutdoorTable",
                "Printer",
                "PS2",
                "PS3",
                "Refrigerator",
                "RoundBed",
                "RoundTable",
                "Sideboard",
                "SideChair",
                "SingleBed",
                "Table",
                "Tank",
                "TvStand",
                "Xbox",
                "Xbox360",
            ],
        )
        # self.scene_manager = SceneManager('./datasets/shapenet_x1.5')
        table_pose = np.eye(4)
        table_pose[:3, 3] = np.array([0.6, 0, -0.25])
        self.scene_manager.set_table_pose(table_pose)
        self.scene_manager.table_bounds = np.array(
            [[0.3, -0.8, 0.001], [1.0, 0.8, 0.001]]
        )
        self.scene_manager.reset()

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

    def check_pairwise_distance(self, mesh_pose):
        coll = False
        for _, (i, j) in enumerate(self.link_combos):
            if abs(i - j) < 2 or ((i, j) == (6, 8)) or ((i, j) == (8, 10)):
                continue
            i_tf = mesh_pose[self.links[i]]
            self.self_collision_managers[i].set_transform("link", i_tf)
            j_tf = mesh_pose[self.links[j]]
            self.self_collision_managers[j].set_transform("link", j_tf)
            coll |= self.self_collision_managers[i].in_collision_other(
                self.self_collision_managers[j]
            )
            if coll:
                return coll
        return coll

    def build_pc_obs(self):
        points = self.scene_manager.render_points()
        points = points[(points[:, :3] > self._hp.bounds[0] + 1e-4).all(axis=1)]
        points = points[(points[:, :3] < self._hp.bounds[1] - 1e-4).all(axis=1)]
        pt_inds = np.random.choice(points.shape[0], size=self._hp.n_points)
        pc = points[pt_inds]
        return pc
