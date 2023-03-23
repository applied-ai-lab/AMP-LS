import os
import numpy as np
import h5py
import trimesh
import gym
import rospy
import moveit_commander
import ros_numpy  # apt install ros-noetic-ros-numpy
from typing import List
import cv2
from cv_bridge import CvBridge, CvBridgeError
from urdfpy import URDF
import itertools
from trimesh.collision import CollisionManager
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import (
    InteractiveMarkerControl,
    Marker,
    InteractiveMarker,
    InteractiveMarkerFeedback,
    InteractiveMarkerUpdate,
    InteractiveMarkerPose,
    MenuEntry,
)
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

from geometry_msgs.msg import *
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.srv import SpawnModel, GetWorldProperties, DeleteModel
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Empty

from core.data.scene_collision.src.scene import SceneManager, SceneRenderer
from core.robots.clients.panda_sim_client import PandaSimClient
from core.utils.general_utils import AttrDict, ParamDict
from core.utils.transform_utils import pose2mat, mat2quat
from core.utils.ros_utils import (
    form_joint_state_msg,
    form_pose_from_ndarray,
    create_marker_message,
    InteractiveMarkerTool,
)
import core.utils.ros_transform_utils as tfu


class GazeboPandaEnv(gym.Env):
    def __init__(self, config=AttrDict()) -> None:
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)

        self.scene_manager = None
        obj_info = h5py.File(
            os.path.join(self._hp.shapenet_datadir, "./object_info.hdf5"), "r"
        )
        self.mesh_info = obj_info["meshes"]
        self.categories = obj_info["categories"]

        # set up node
        rospy.init_node("gazebo_panda_env", anonymous=True)
        self.robot_client = PandaSimClient(self._hp.robot_client_params)

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

        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)

        # connect to state validity service client
        rospy.wait_for_service("/check_state_validity")
        self.moveit_sv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")

        # connect to clear octomap service client
        rospy.wait_for_service("/clear_octomap")
        self.clear_octomap = rospy.ServiceProxy("/clear_octomap", Empty)
        rospy.loginfo("Connecting to Clear Octomap service")

        # point cloud subscriber
        # rospy.Subscriber("/camera/depth_registered/points", PointCloud2, callback=self._get_pc_cb, queue_size=1)
        rospy.Subscriber(
            "/move_group/filtered_cloud",
            PointCloud2,
            callback=self._get_pc_cb,
            queue_size=1,
        )
        self.pc = None

        # camera observation subscriber
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            callback=self._get_rgb_img_cb,
            queue_size=1,
        )
        self.rgb_img = None
        rospy.Subscriber(
            "/camera/depth_registered/image_raw",
            Image,
            callback=self._get_depth_img_cb,
            queue_size=1,
        )
        self.depth_img = None

        self.bridge = CvBridge()

        # gazebo spawn object
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model_client = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel
        )

        self.get_world_property = rospy.ServiceProxy(
            "/gazebo/get_world_properties", GetWorldProperties
        )
        self.delete_model_client = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel
        )

        self.marker_pub = rospy.Publisher(
            "rviz_visual_tools", MarkerArray, queue_size=1
        )
        self.interactive_tool = InteractiveMarkerTool()
        self.spawn_table()

    def _default_hparams(self):
        default_params = ParamDict(
            {
                "shapenet_datadir": os.path.join(
                    os.environ["DATA_DIR"], "./shapenet_x1.5"
                ),
                "robot_urdf": "./core/data/scene_collision/assets/data/panda/panda.urdf",
                "max_num_objs": 10,
                "min_num_objs": 6,
                "n_points": 8192,  #  number of points for point cloud
                "bg_seg_label": 0,
                "env_seg_label": 1,
                "robot_seg_label": 2,
                "target_seg_label": 3,
                "robot_client_params": AttrDict(),
                "robot_to_model": (-0.6, 0, 0.25),
                "bounds": None,
                "moving_goal": False,
                "arrange_mesh_scene": True,
            }
        )
        return default_params

    def _get_pc_cb(self, data):
        self.pc = data

    def _get_rgb_img_cb(self, data):
        self.rgb_img = data

    def _get_depth_img_cb(self, data):
        self.depth_img = data

    def check_for_collision(self, robot_state: RobotState):
        """
        Check for self collision or collision of the panda arm with other collision objects.
        """
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = "panda_arm"
        result = self.moveit_sv.call(gsvr)
        return not result.valid

    def has_collision(self, joints):
        robot_to_model = np.eye(4)
        # robot_to_model[:3, 3] = (-0.6, 0, 0.25)
        link_pose = self.robot.link_fk(np.concatenate([joints, np.array([0.04])]))
        coll = False
        for link in self.links[1:]:
            pose = link_pose[link]
            self.scene_manager._collision_manager.add_object(
                "query_object", link.collision_mesh, robot_to_model @ pose
            )
            coll = coll or self.scene_manager.collides()
            self.scene_manager._collision_manager.remove_object("query_object")
            if coll:
                return coll
        return coll

    def sample_valid_jpos_xyz(self, bounds=None, move=False) -> List[float]:
        """
        The configuration is stored as a Numpy array with one row in the format
        [j1, j2, ..., j7, ee_x, ee_y, ee_z].
        Cases where there is self-collision or where the end-effector is under the table are discarded.
        """
        robot_to_model = np.eye(4)
        # robot_to_model[:3, 3] = self._hp.robot_to_model

        while True:
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
        if move:
            self.robot_client.set_joint_state(rand_cfg[:7])
        return np.concatenate([rand_cfg[:7], ee_pose[:3, 3], mat2quat(ee_pose[:3, :3])])

    def clear_objects(self):
        for model_name in self.get_world_property().model_names:
            if "obj" in model_name:
                self.delete_model_client(model_name)

    def arrange_objects(self):
        self.clear_objects()
        n_objs = np.random.randint(self._hp.min_num_objs, self._hp.max_num_objs)

        robot_to_model = np.eye(4)
        pose = np.eye(4)
        self._create_scene()
        self.scene_manager._collision_manager.add_object(
            "query_object", self.links[0].collision_mesh, robot_to_model @ pose
        )
        self.scene_manager.arrange_scene(n_objs)
        # import pyrender
        # pyrender.viewer.Viewer(self.scene_manager._renderer._scene, viewer_flags={'use_direct_lighting': True, 'lighting_intensity': 1.0})
        self.scene_manager._collision_manager.remove_object("query_object")

        if len(self.scene_manager.objs) == 0:
            raise ValueError("Environment does not have any objects!")

        scene_info = []

        for i, (obj_name, obj_info) in enumerate(self.scene_manager.objs.items()):
            if obj_name == "table":
                continue
            else:
                self.spawn_model_client(
                    model_name="obj_{:d}".format(i),
                    model_xml=open(
                        os.path.join(
                            self._hp.shapenet_datadir,
                            obj_info["path"].replace("obj", "sdf"),
                        ),
                        "r",
                    ).read(),
                    initial_pose=form_pose_from_ndarray(obj_info["pose"]),
                    reference_frame="world",
                )

                obj_output = {}
                obj_output["id"] = i
                obj_output["name"] = obj_name
                obj_output["pose"] = obj_info[
                    "pose"
                ].tolist()  # ndarray is not serializable when saving as json
                obj_output["path"] = obj_info["path"]

                scene_info.append(obj_output)

        return scene_info

    def _create_scene(self):
        # scene manager
        if self.scene_manager is not None:
            del self.scene_manager
        r = SceneRenderer()
        # self.scene_manager = SceneManager(self._hp.shapenet_datadir, renderer=r)
        self.scene_manager = SceneManager(
            self._hp.shapenet_datadir,
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
                "Computer",
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
        table_pose = np.eye(4)
        table_pose[:3, 3] = np.array([0.6, 0, -0.25])
        self.scene_manager.set_table_pose(table_pose)
        self.scene_manager.table_bounds = np.array(
            [[0.3, -0.8, 0.001], [1.0, 0.8, 0.001]]
        )
        self.scene_manager.reset()

    def arrange_objects_to(self, scene_info):
        self.clear_objects()
        self._create_scene()
        self.scene_manager.arrange_table()

        for obj_input in scene_info:
            obj_id = "obj_{:d}".format(obj_input["id"])

            mesh_path = os.path.join(self._hp.shapenet_datadir, obj_input["path"])
            mesh = trimesh.load(mesh_path, force="mesh")

            pose = np.array(obj_input["pose"])  # convert list back to ndarray

            if self._hp.arrange_mesh_scene:
                self.scene_manager.add_object(name=obj_id, mesh=mesh, pose=pose)

            self.spawn_model_client(
                model_name=obj_id,
                model_xml=open(
                    os.path.join(
                        self._hp.shapenet_datadir,
                        obj_input["path"].replace("obj", "sdf"),
                    ),
                    "r",
                ).read(),
                initial_pose=form_pose_from_ndarray(pose),
                reference_frame="world",
            )

    # GymEnv functions
    def reset(self):
        """Resets all internal variables of the environment."""
        self._create_scene()
        self.arrange_objects()
        rospy.sleep(0.5)

        # run `roslaunch panda_moveit_config *camera_connected.launch` to sample valid joint states
        goal_states = self.sample_valid_jpos_xyz(bounds=self._hp.bounds)
        self.goal = goal_states[7:10]
        self.goal_ori = goal_states[-4:]
        self.goal_joints = goal_states[:7]

        # Rviz visualisation
        if self._hp.moving_goal:
            pose = Point(self.goal[0], self.goal[1], self.goal[2])
            self.interactive_tool.make6DofMarker(
                InteractiveMarkerControl.MOVE_3D, pose, "panda_link0", False
            )
            self.interactive_tool.server.applyChanges()
        else:
            goal_marker_msg = create_marker_message(
                1, self.goal[0], self.goal[1], self.goal[2], (1.0, 0.0, 0.0, 1.0)
            )
            marker_array_msg = MarkerArray()
            marker_array_msg.markers.append(goal_marker_msg)
            self.marker_pub.publish(marker_array_msg)

        start_states = self.sample_valid_jpos_xyz(bounds=self._hp.bounds, move=True)

        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = self.robot_client.get_current_state().joint_state.position
        robot_joint_state.append(
            0.0
        )  # temporarily append static fingertip state TODO: modify this later
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            robot_states=np.array(robot_joint_state),
            goal=self.goal,
            goal_ori=self.goal_ori,
            goal_joints=self.goal_joints,
        )
        obs.update(pc)
        return obs

    def reset_to(self, scene_config):
        # retrieve scene info from init config
        scene_info = scene_config["scene_info"]
        ee_jpos_xyz = np.array(scene_config["ee_jpos_xyz"])
        goal_jpos_xyz = np.array(scene_config["goal_jpos_xyz"])
        jpos = ee_jpos_xyz[:7]
        ee_x, ee_y, ee_z = ee_jpos_xyz[7:10]
        g_x, g_y, g_z = goal_jpos_xyz[7:10]
        goal_quat = goal_jpos_xyz[10:]

        self.goal = goal_jpos_xyz[7:10]
        self.goal_ori = goal_jpos_xyz[10:]
        self.goal_joints = goal_jpos_xyz[:7]

        self.clear_objects()
        self.delete_model_client("table")
        self.robot_client.set_joint_state(jpos)
        robot_joint_state = jpos

        self.spawn_table()
        rospy.sleep(0.2)
        self.arrange_objects_to(scene_info)

        # Rviz visualisation
        ee_marker_msg = create_marker_message(0, ee_x, ee_y, ee_z, (0.0, 1.0, 0.0, 1.0))
        goal_marker_msg = create_marker_message(1, g_x, g_y, g_z, (1.0, 0.0, 0.0, 1.0))
        marker_array_msg = MarkerArray()
        marker_array_msg.markers.append(ee_marker_msg)
        marker_array_msg.markers.append(goal_marker_msg)
        self.marker_pub.publish(marker_array_msg)

        self.clear_octomap()
        rospy.sleep(2.0)
        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = ee_jpos_xyz[:7]
        robot_joint_state = np.concatenate([robot_joint_state, np.array([0.04])])
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            robot_states=np.array(robot_joint_state),
            goal=self.goal,
            goal_ori=self.goal_ori,
            goal_joints=self.goal_joints,
        )
        obs.update(pc)
        return obs

    def step(self, action):
        """Performs one environment step. Returns dict <next observation, reward, done, info>."""
        # next_joint_state = self.robot_client.get_current_state().joint_state.position
        # next_joint_state += action
        # self.robot_client.set_joint_state(next_joint_state)
        self.robot_client.set_joint_state(action)

        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = self.robot_client.get_current_state().joint_state.position
        robot_joint_state.append(
            0.04
        )  # temporarily append static fingertip state TODO: modify this later
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            robot_states=np.array(robot_joint_state),
            goal=self.goal,
            goal_ori=self.goal_ori,
            goal_joints=self.goal_joints,
        )
        obs.update(pc)

        reward = 0.0  # TODO: replace this with distance
        done = False  # TODO: replace this
        info = AttrDict()

        return obs, reward, done, info

    def get_obs_as_list(self, scene_config):
        # retrieve scene info from init config
        ee_jpos_xyz = np.array(scene_config["ee_jpos_xyz"])
        goal_jpos_xyz = np.array(scene_config["goal_jpos_xyz"])

        self.goal = goal_jpos_xyz[7:10]
        self.goal_ori = goal_jpos_xyz[10:]
        self.goal_joints = goal_jpos_xyz[:7]

        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = ee_jpos_xyz[:7]
        robot_joint_state = np.concatenate([robot_joint_state, np.array([0.04])])
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot.tolist(),
            robot_to_model=rtm.tolist(),
            model_to_robot=np.linalg.inv(rtm).tolist(),
            robot_states=np.array(robot_joint_state).tolist(),
            goal=self.goal.tolist(),
            goal_ori=self.goal_ori.tolist(),
            goal_joints=self.goal_joints.tolist(),
        )
        obs.update(pc)
        obs["pc"] = obs["pc"].tolist()
        obs["pc_label"] = obs["pc_label"].tolist()
        del obs["depth_img"]
        # obs['depth_img'] = obs['depth_img'].tolist()

        return obs

    def render(self, mode="rgb_array"):
        camera_obs = self.get_camera_obs()
        return camera_obs["color"]

    def get_camera_obs(self):
        while self.rgb_img is None or self.depth_img is None:
            rospy.sleep(0.1)

        try:
            cv_rgb_image = self.bridge.imgmsg_to_cv2(
                self.rgb_img, self.rgb_img.encoding
            )  # encoding: rgb8; shape: (480, 640, 3)

            cv_depth_image = self.bridge.imgmsg_to_cv2(
                self.depth_img, self.depth_img.encoding
            )  # encoding: 32FC1; shape: (480, 640)
            # normalise the depth image to fall between 0 (black) and 1 (white)
            cv_depth_image = cv2.normalize(
                cv_depth_image, cv_depth_image, 0, 1, cv2.NORM_MINMAX
            )
            # resize to the desired size
            # cv_depth_image = cv2.resize(cv_depth_image, desired_shape, interpolation=cv2.INTER_CUBIC)
        except CvBridgeError as e:
            print(e)

        # visualise images
        # cv2.imshow("RBG image", cv_rgb_image)
        # cv2.waitKey(0)
        # cv2.imshow("Depth image", cv_depth_image)
        # cv2.waitKey(0)

        camera_data = {"color": cv_rgb_image, "depth": cv_depth_image}

        return camera_data

    def get_pc_obs(self):
        camera_obs = self.get_camera_obs()
        while self.pc is None:
            rospy.sleep(0.1)

        pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
            self.pc, remove_nans=True
        )  # shape: (480 * 640, 3)

        camera_pose = tfu.current_robot_pose("world", "camera_link")
        camera_mat = pose2mat(
            [
                tfu.position_to_list(camera_pose.position),
                tfu.quaternion_to_list(camera_pose.orientation),
            ]
        )

        pc_4d_np = np.concatenate((pc_np, np.ones((pc_np.shape[0], 1))), axis=-1)
        pc_4d_np = np.transpose(np.dot(camera_mat, np.transpose(pc_4d_np)))
        pc_np = pc_4d_np[:, :3]

        cam_valid_index = pc_np[:, 2] > -0.01
        valid_index = np.where(cam_valid_index)[0]
        # sample points
        mask = np.random.choice(
            valid_index,
            size=self._hp.n_points,
            replace=len(valid_index) < self._hp.n_points,
        )
        pc_np = pc_np[mask, :]

        # visualise point cloud
        # pc = trimesh.PointCloud(np.array(pc_np))
        # pc.show()

        # visualise point cloud using PCL
        # https://github.com/strawlab/python-pcl/blob/master/examples/visualization.py
        # pc_pcl = pcl.PointCloud(np.array(pc_np, dtype=np.float32))
        # visual = pcl.pcl_visualization.CloudViewing()
        # visual.ShowMonochromeCloud(pc_pcl, b'cloud')
        # v = True
        # while v:
        #     v = not(visual.WasStopped())

        return AttrDict(
            pc=pc_np,
            pc_label=np.ones(len(pc_np)),
            depth_img=camera_obs["depth"],
            label_map=AttrDict(
                robot=self._hp.robot_seg_label,
                env=self._hp.env_seg_label,
                bg=self._hp.bg_seg_label,
                target=self._hp.target_seg_label,
            ),
        )

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

    def add_box2scene(self, name, position, orientation, size):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.robot_client.panda_arm.get_planning_frame()
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        pose_stamped.pose.orientation.x = orientation[0]
        pose_stamped.pose.orientation.y = orientation[1]
        pose_stamped.pose.orientation.z = orientation[2]
        pose_stamped.pose.orientation.w = orientation[3]
        self.scene.add_box(name, pose_stamped, size)

    def spawn_table(self):
        table_pose = np.eye(4)
        table_pose[:3, 3] = [0.6, 0, -0.1]
        self.spawn_model_client(
            model_name="table",
            model_xml=open("./core/data/gazebo/assets/models/table.sdf").read(),
            initial_pose=form_pose_from_ndarray(table_pose),
            reference_frame="world",
        )


class GazeboPandaBlockEnv(GazeboPandaEnv):
    def arrange_objects(self):
        self._create_scene()
        self.scene_manager.arrange_fixed_scene(offset=0.6)

    def reset(self):
        """Resets all internal variables of the environment."""
        self._create_scene()
        self.arrange_objects()
        rospy.sleep(0.5)

        # run `roslaunch panda_moveit_config *camera_connected.launch` to sample valid joint states
        goal_states = self.sample_goal_jpos_xyz()
        self.goal = goal_states[7:10]
        self.goal_ori = goal_states[-4:]
        self.goal_joints = goal_states[:7]

        # Rviz visualisation
        if self._hp.moving_goal:
            pose = Point(self.goal[0], self.goal[1], self.goal[2])
            self.interactive_tool.make6DofMarker(
                InteractiveMarkerControl.MOVE_3D, pose, "panda_link0", False
            )
            self.interactive_tool.server.applyChanges()
        else:
            goal_marker_msg = create_marker_message(
                1, self.goal[0], self.goal[1], self.goal[2], (1.0, 0.0, 0.0, 1.0)
            )
            marker_array_msg = MarkerArray()
            marker_array_msg.markers.append(goal_marker_msg)
            self.marker_pub.publish(marker_array_msg)

        start_states = self.sample_start_jpos_xyz(move=True)

        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = self.robot_client.get_current_state().joint_state.position
        robot_joint_state.append(
            0.0
        )  # temporarily append static fingertip state TODO: modify this later
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            robot_states=np.array(robot_joint_state),
            goal=self.goal,
            goal_ori=self.goal_ori,
            goal_joints=self.goal_joints,
        )
        obs.update(pc)
        return obs

    def sample_start_jpos_xyz(self, move=False) -> List[float]:
        """
        The configuration is stored as a Numpy array with one row in the format
        [j1, j2, ..., j7, ee_x, ee_y, ee_z].
        Cases where there is self-collision or where the end-effector is under the table are discarded.
        """
        robot_to_model = np.eye(4)
        joints = np.array(
            [
                -0.004247765649998847,
                -0.022489946792044968,
                -0.15420333925431695,
                -2.8125067467857683,
                -0.009477623621254594,
                2.79237164716632,
                0.6358937293218005,
                0.04,
            ]
        )

        ee_pose = self.robot.link_fk(joints, link="panda_link8")
        if move:
            self.robot_client.set_joint_state(joints[:7])
        return np.concatenate([joints[:7], ee_pose[:3, 3], mat2quat(ee_pose[:3, :3])])

    def sample_goal_jpos_xyz(self) -> List[float]:
        """
        The configuration is stored as a Numpy array with one row in the format
        [j1, j2, ..., j7, ee_x, ee_y, ee_z].
        Cases where there is self-collision or where the end-effector is under the table are discarded.
        """
        robot_to_model = np.eye(4)
        joints = np.array(
            [
                -0.9773798551759905,
                -1.1773490618264555,
                1.8395873950508648,
                -1.984403758998667,
                0.9728730606353828,
                2.0925241756797686,
                1.2173386779676978,
                0.04,
            ]
        )

        ee_pose = self.robot.link_fk(joints, link="panda_link8")
        return np.concatenate([joints[:7], ee_pose[:3, 3], mat2quat(ee_pose[:3, :3])])


class GazeboPandaTestEnv(GazeboPandaEnv):
    def _create_scene(self):
        # scene manager
        if self.scene_manager is not None:
            del self.scene_manager
        r = SceneRenderer()
        self.scene_manager = SceneManager(
            self._hp.shapenet_datadir,
            renderer=r,
            cat=[
                "1Shelves",
                "2Shelves",
                "3Shelves",
                "4Shelves",
                "5Shelves",
                "6Shelves",
                "7Shelves",
                "Cabinet",
                "Chair",
                "ChestOfDrawers",
                "ChildBed",
                "CoffeeTable",
                "Desk",
                "Desktop",
                "DiningTable",
                "DoubleBed",
                "KneelingChair",
                "Microwave",
                "OfficeChair",
                "OfficeSideChair",
                "Refrigerator",
                "SingleBed",
                "StandingClock",
                "Table",
            ],
        )
        table_pose = np.eye(4)
        table_pose[:3, 3] = np.array([0.6, 0, -0.25])
        self.scene_manager.set_table_pose(table_pose)
        self.scene_manager.table_bounds = np.array(
            [[0.3, -0.8, 0.001], [1.0, 0.8, 0.001]]
        )
        self.scene_manager.reset()


class GazeboPandaConveyorEnv(GazeboPandaEnv):
    def __init__(self, config=AttrDict()) -> None:
        super().__init__(config)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._update_goal)
        self.goal = None
        self.goal_ori = None

    def _update_goal(self, msg):
        for i, name in enumerate(msg.name):
            if "red_cylinder" in name:
                pos = msg.pose[i].position
                self.goal = np.array([pos.x, pos.y, pos.z + 0.05])
                self.goal_ori = np.array(
                    [
                        0.923908504857489,
                        -0.382612813753092,
                        0.0006593793134512882,
                        -0.00027317185331890274,
                    ]
                )

    # GymEnv functions
    def reset(self):
        if not self._hp.moving_goal:
            goal_marker_msg = create_marker_message(
                1, self.goal[0], self.goal[1], self.goal[2], (1.0, 0.0, 0.0, 1.0)
            )
            marker_array_msg = MarkerArray()
            marker_array_msg.markers.append(goal_marker_msg)
            self.marker_pub.publish(marker_array_msg)

        start_states = self.sample_valid_jpos_xyz(bounds=self._hp.bounds, move=True)

        pc = self.get_pc_obs()
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        robot_joint_state = np.concatenate([start_states[:7], np.array([0.04])])
        world_to_robot = np.eye(4)
        obs = AttrDict(
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            robot_states=np.array(robot_joint_state),
            goal=self.goal,
            goal_ori=self.goal_ori,
        )
        obs.update(pc)
        return obs


if __name__ == "__main__":
    # env = GazeboPandaEnv()
    # env = GazeboPandaBlockEnv()
    env = GazeboPandaConveyorEnv()
    # while True:
    #     rospy.sleep(0.1)
    # env.reset()
    # scene_info = env.arrange_objects()
    # env.arrange_objects_to(scene_info)
    # env.get_pc_obs()
