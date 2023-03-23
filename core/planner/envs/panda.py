import os, sys

import gym
import cv2
import numpy as np
from math import ceil
import trimesh
import rospy
import time
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Image, PointCloud2
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
import moveit_msgs.srv
import ros_numpy
from cv_bridge import CvBridge, CvBridgeError
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
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from core.utils.transform_utils import (
    pose2mat,
    mat2quat,
    quat2mat,
    pose_in_A_to_pose_in_B,
    make_pose,
)
from core.utils.general_utils import ParamDict, AttrDict
from core.utils.ros_utils import InteractiveMarkerTool, form_joint_state_msg
from core.robots.clients.panda_client import PandaClient
from core.utils.eval_utils import mean_angle_btw_vectors
import core.utils.ros_transform_utils as tfu
from std_srvs.srv import Empty
import h5py

KEYS = ["action", "observation", "reward"]


def _get_rotated(R):
    INIT_AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if len(R.shape) < 3:
        R = np.expand_dims(R, axis=0)
    return np.transpose(np.matmul(R, np.transpose(INIT_AXES)), axes=[0, 2, 1])


class PandaEnv(gym.Env):
    def __init__(self, config=AttrDict()):
        super().__init__()
        self._hp = self._default_hparams().overwrite(config)
        rospy.init_node("panda_env", anonymous=True)
        self._cv_bridge = CvBridge()

        self.rate = rospy.Rate(1.0 / self._hp.dt)  # apply policy at 1.0/dt Hz
        client_config = AttrDict(
            home_pose_joint_values=self._hp.home_pose_joint_values,
            ee_safety_zone=self._hp.ee_safety_zone,
        )
        self._hp.client_config.update(client_config)
        self.panda_client = PandaClient(self._hp.client_config)
        # self.panda_client.set_collision_behavior()
        self._success = False
        self.construct_scene()

    def get_joint_states(self):
        pass

    def set_joint_states(self, joint_states):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "dt": 0.2,
                "client_config": AttrDict(load_moveit=True),
                "from_pixels": False,
                "reward_type": "dense",
            }
        )
        return default_dict

    def render(self, mode="rgb_array"):
        # resized_color_img = cv2.resize(self._color_img, (320, 240))
        resized_color_img = np.zeros((240, 240, 3)).astype(np.uint8)
        return resized_color_img

    def compute_distance(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal.reshape(1, -1)
        desired_goal = desired_goal.reshape(1, -1)
        distance = np.sqrt(np.sum(np.square(achieved_goal - desired_goal), axis=1))
        return float(distance)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def construct_scene(self):
        # self.panda_client.add_box2scene('table', [0.655, 0, -0.05], [0, 0, 0, 1.0], (1.0, 0.9, 0.1))
        self.panda_client.add_box2scene(
            "back_wall", [-0.3, 0, 0.5], [0, 0, 0, 1.0], (0.1, 1.0, 0.9)
        )
        # self.panda_client.add_box2scene('right_wall', [0.655, 0.6, 0.5], [0, 0, 0, 1.0], (1.0, 0.1, 1.0))
        # self.panda_client.add_box2scene('left_wall', [0.655, -0.6, 0.5], [0, 0, 0, 1.0], (1.0, 0.1, 1.0))
        # self.panda_client.add_box2scene('top', [0.655, 0, 1.1], [0, 0, 0, 1.0], (1.0, 0.9, 0.1))


class PandaReachEnv(PandaEnv):
    def __init__(self, config=AttrDict()):
        super().__init__(config)
        self.interactive_tool = InteractiveMarkerTool()

        # connect to state validity service client
        rospy.wait_for_service("/check_state_validity")
        self.moveit_sv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")

        rospy.Subscriber(
            "/move_group/filtered_cloud", PointCloud2, callback=self._get_pc_cb
        )
        self.pc = None
        # # camera observation subscriber
        # rospy.Subscriber("/camera/color/image_raw", Image, callback=self._get_rgb_img_cb)
        # self.rgb_img = None
        rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            callback=self._get_depth_img_cb,
        )
        self.depth_img = None

        # connect to clear octomap service client
        rospy.wait_for_service("/clear_octomap")
        self.clear_octomap = rospy.ServiceProxy("/clear_octomap", Empty)
        rospy.loginfo("Connecting to Clear Octomap service")

        self.bridge = CvBridge()

        self.goal_pos_list = []
        self.goal_rot_list = []
        self.current_joint_list = []

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "dt": 0.1,
                "threshold": 0.05,
                "home_pose_joint_values": [
                    0.0,
                    0.19634954,
                    0.0,
                    -2.61799388,
                    0.0,
                    2.94159265,
                    0.78539816,
                ],
                "ee_safety_zone": [[0.30, 0.76], [-0.18, 0.18], [0.01, 0.4]],
                "moving_goal": False,
                "n_points": 8192,  #  number of points for point cloud
                "bg_seg_label": 0,
                "env_seg_label": 1,
                "robot_seg_label": 2,
                "target_seg_label": 3,
                "robot_client_params": AttrDict(),
                "robot_to_model": (-0.6, 0, 0.25),
                # 'robot_to_model': (-0.6, 0, 0.2),
                "bounds": None,
                "goal": None,
                "goal_quat": None,
                "use_tag_goal": False,
                "tag_id": None,
                "record_traj": True,
                "record_traj_suffix": "",
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def _get_pc_cb(self, data):
        self.pc = data

    def _get_rgb_img_cb(self, data):
        self.rgb_img = data

    def _get_depth_img_cb(self, data):
        self.depth_img = data

    def reset(self):
        self.clear_octomap()
        # self.goal = np.random.uniform([0.45, -0.3, 0.4], [0.7, 0.3, 0.5])
        if self._hp.use_tag_goal:
            marker_pose = tfu.current_robot_pose("panda_link0", self._hp.tag_id)
            if marker_pose is None:
                marker_pose = self.marker_pose
            self.marker_pose = marker_pose
            self.goal = np.array(
                [
                    marker_pose.position.x,
                    marker_pose.position.y,
                    marker_pose.position.z + 0.25,
                ]
            )
            if self._hp.goal_quat is None:
                raise NotImplementedError
            else:
                self.goal_rot = quat2mat(self._hp.goal_quat)
        elif self._hp.goal is None or self._hp.goal_quat is None:
            bounds = np.array([[0.2, -0.5, 0.2], [0.6, 0.5, 0.4]])
            goal_joint = self.sample_valid_joint(bounds=bounds)
            goal_ee_pose = self.panda_client.get_ee_pose(goal_joint)
            self.goal = goal_ee_pose[:3, 3]
            self.goal_rot = goal_ee_pose[:3, :3]
        else:
            self.goal = self._hp.goal
            self.goal_rot = quat2mat(self._hp.goal_quat)

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.panda_client.panda_arm.get_planning_frame()
        goal_pose.pose.position.x = self.goal[0]
        goal_pose.pose.position.y = self.goal[1]
        goal_pose.pose.position.z = self.goal[2]
        if self._hp.moving_goal:
            pose = Point(self.goal[0], self.goal[1], self.goal[2])
            quat = mat2quat(self.goal_rot)
            orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
            self.interactive_tool.make6DofMarker(
                InteractiveMarkerControl.MOVE_3D,
                pose,
                "panda_link0",
                show_6dof=True,
                scale=0.15,
                orientation=orientation,
            )
            self.interactive_tool.server.applyChanges()
            ready = False
            while not ready:
                ans = input("Are you ready?[Yes/No]\n")
                if ans.lower() == "yes" or ans.lower() == "y":
                    ready = True

        else:
            self.panda_client.scene.add_sphere(name="goal", pose=goal_pose, radius=0.02)
            # self.panda_client.scene.set_color(name='goal', r=1, g=0, b=0)
        obs = self._get_obs()
        self._success = False

        self.goal_pos_list.append(self.goal)
        self.goal_rot_list.append(self.goal_rot)
        self.current_joint_list.append(obs["robot_states"])
        return obs

    def step(self, action):
        if action.any():
            self.panda_client.set_joint_pos(action)
            # print(action)
            self.rate.sleep()

        obs = self._get_obs()
        achieved_goal_pose = self.panda_client.get_ee_pose(action)
        achieved_goal = achieved_goal_pose[:3, 3]
        goal = obs["goal"]
        reward = 0
        # reward = self._compute_reward(achieved_goal, goal)

        self.distance = self.compute_distance(achieved_goal, goal)
        done = False
        self._success = self.distance < self._hp.threshold

        info = AttrDict(distance=self.distance)
        self.goal_pos_list.append(obs["goal"])
        self.goal_rot_list.append(obs["goal_rot"])
        self.current_joint_list.append(obs["robot_states"])
        return obs, np.array(reward), np.array(done), info

    def _get_obs(self):
        joint_states = self.panda_client.current_joints[:7]
        joint_states = np.concatenate([joint_states, np.array([0.04])])
        # ee_pose = self.panda_client.get_ee_pose(joint_states)
        # ee_pos = ee_pose[:3, 3]
        # obs = np.concatenate((joint_states, ee_pos))
        if self._hp.moving_goal:
            goal = np.array(self.interactive_tool.marker_pos)
            goal_rot = quat2mat(np.array(self.interactive_tool.marker_quat))
        else:
            goal = self.goal
            goal_rot = self.goal_rot

        world_to_robot = np.eye(4)
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        pc = self.get_pc_obs()
        obs = AttrDict(
            robot_states=joint_states,
            goal=goal,
            goal_rot=goal_rot,
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            ignore_coll=False,
        )
        obs.update(pc)
        return obs

    def _compute_reward(self, achieved_goal, desired_goal):
        distance = self.compute_distance(achieved_goal, desired_goal)
        if self._hp.reward_type == "dense":
            return -distance
        else:
            return -float(distance >= self._hp.threshold)

    def get_episode_info(self):
        with h5py.File(
            "./records/online_planning_traj_{}.h5".format(self._hp.record_traj_suffix),
            "w",
        ) as F:
            F["goal_pos"] = self.goal_pos_list
            F["goal_rot"] = self.goal_rot_list
            F["current_joints"] = self.current_joint_list
        return AttrDict(success=float(self._success), distance=self.distance)

    def get_camera_obs(self):
        # cv_rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_img, self.rgb_img.encoding)  # encoding: rgb8; shape: (480, 640, 3)
        cv_depth_image = self.bridge.imgmsg_to_cv2(
            self.depth_img, self.depth_img.encoding
        )  # encoding: 32FC1; shape: (480, 640)
        # normalise the depth image to fall between 0 (black) and 1 (white)
        cv_depth_image = cv2.normalize(
            cv_depth_image, cv_depth_image, 0, 1, cv2.NORM_MINMAX
        )
        # resize to the desired size
        # cv_depth_image = cv2.resize(cv_depth_image, desired_shape, interpolation=cv2.INTER_CUBIC)

        # visualise images
        # cv2.imshow("RBG image", cv_rgb_image)
        # cv2.waitKey(0)
        # cv2.imshow("Depth image", cv_depth_image)
        # cv2.waitKey(0)

        # camera_data = {'color': cv_rgb_image, 'depth': cv_depth_image}
        camera_data = {"depth": cv_depth_image}

        return camera_data

    def get_pc_obs(self):
        while self.depth_img is None:
            print("sleep")
            rospy.sleep(0.1)
        camera_obs = self.get_camera_obs()

        while self.pc is None:
            print("sleep")
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

        # cam_valid_index = pc_np[:, 2] > -0.0
        cam_valid_index = pc_np[:, 2] > -0.1
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

    def sample_valid_joint(self, bounds=None):
        """
        The configuration is stored as a Numpy array with one row in the format
        [j1, j2, ..., j7, ee_x, ee_y, ee_z].
        Cases where there is self-collision or where the end-effector is under the table are discarded.
        """
        while True:
            # sample a random joint configuration
            rand_cfg = (
                np.random.rand(len(self.panda_client.low_joint_vals))
                * (self.panda_client.high_joint_vals - self.panda_client.low_joint_vals)
                + self.panda_client.low_joint_vals
            )

            ee_pose = self.panda_client.kinematics.link_fk(rand_cfg, link="panda_link8")
            if bounds is not None:
                ee_pos = ee_pose[:3, 3]
                in_bound = (ee_pos > bounds[0] + 1e-5).all()
                in_bound &= (ee_pos < bounds[1] - 1e-5).all()
                if not in_bound:
                    continue
            # self_coll = self.check_pairwise_distance(link_pose)
            # if self_coll:
            #     continue

            robot_state = RobotState()
            robot_state.joint_state = form_joint_state_msg(rand_cfg[:7])
            coll = self.check_for_collision(robot_state)
            if not coll:
                break

        return rand_cfg[:7]

    def check_for_collision(self, robot_state: RobotState):
        """
        Check for self collision or collision of the panda arm with other collision objects.
        """
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = "panda_arm"
        result = self.moveit_sv.call(gsvr)
        return not result.valid


class PandaTableReachEnv(PandaReachEnv):
    def reset(self):
        self.clear_octomap()
        rospy.sleep(0.1)
        marker_pose = tfu.current_robot_pose("panda_link0", self._hp.tag_id)
        marker_rot = make_pose(
            np.array(
                [marker_pose.position.x, marker_pose.position.y, marker_pose.position.z]
            ),
            quat2mat(
                np.array(
                    [
                        marker_pose.orientation.x,
                        marker_pose.orientation.y,
                        marker_pose.orientation.z,
                        marker_pose.orientation.w,
                    ]
                )
            ),
        )
        target_pose_in_marker_frame = make_pose(
            np.array([-0.25, 0.02, 0.21]),
            quat2mat(np.array([0.9299, -0.3674, -0.0032, 0.01565])),
        )
        target_pose = pose_in_A_to_pose_in_B(target_pose_in_marker_frame, marker_rot)

        self.goal = target_pose[:3, 3]
        self.goal_rot = target_pose[:3, :3]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.panda_client.panda_arm.get_planning_frame()
        goal_pose.pose.position.x = self.goal[0]
        goal_pose.pose.position.y = self.goal[1]
        goal_pose.pose.position.z = self.goal[2]

        # self.panda_client.scene.add_cylinder(name='target', pose=marker_pose,   radius=0.015)
        self.panda_client.scene.add_sphere(name="goal", pose=goal_pose, radius=0.02)
        obs = self._get_obs()
        self._success = False
        self.flag = 0
        return obs

    def step(self, action):
        if action.any():
            self.panda_client.set_joint_pos(action)
            print(action)
            self.rate.sleep()

        obs = self._get_obs()
        achieved_goal_pose = self.panda_client.get_ee_pose(action)
        achieved_goal = achieved_goal_pose[:3, 3]
        goal = obs["goal"]

        reward = 0
        self.distance = self.compute_distance(achieved_goal, goal)
        angle = mean_angle_btw_vectors(
            _get_rotated(achieved_goal_pose[:3, :3]), _get_rotated(obs["goal_rot"])
        )

        print("distance: ", self.distance, " angle: ", angle)
        if self.distance < 0.015 and angle < 15 and self.flag == 0:
            self.goal[2] -= 0.9
            self.flag = 1
        elif self.distance < 0.01 and self.flag == 1:
            self.panda_client.gripper_grasp(0.04, force=70)
            self.goal[2] += 0.1
            self.flag = 2

        if self.flag > 0:
            obs.update(AttrDict(ignore_coll=True))
        done = False
        self._success = self.distance < self._hp.threshold

        info = AttrDict(distance=self.distance)
        return obs, np.array(reward), np.array(done), info


class PandaConveyorReachEnv(PandaReachEnv):
    def __init__(self, config=AttrDict()):
        super().__init__(config)
        self.tag_detection = rospy.Subscriber(
            "/tag_detections", AprilTagDetectionArray, callback=self._tag_detection
        )
        self.ids = []
        self.sizes = []
        self.marker_found = False
        self.missing_marker = False

    def _tag_detection(self, msg):
        self.ids = []
        for detection in msg.detections:
            self.ids.append(detection.id[0])

    def reset(self):
        self.clear_octomap()

        target_pose, marker_found = self.get_target_pose()
        self.goal = target_pose[:3, 3]
        self.goal_rot = target_pose[:3, :3]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.panda_client.panda_arm.get_planning_frame()
        goal_pose.pose.position.x = self.goal[0]
        goal_pose.pose.position.y = self.goal[1]
        goal_pose.pose.position.z = self.goal[2]
        self.panda_client.scene.add_sphere(name="goal", pose=goal_pose, radius=0.02)

        obs = self._get_obs()
        self._success = False
        self.flag = 0
        return obs

    def get_target_pose(self):
        new_marker_found = False
        if self._hp.use_tag_goal:
            if int(self._hp.tag_id.split("_")[1]) in self.ids:
                marker_pose = tfu.current_robot_pose("panda_link0", self._hp.tag_id)
                if marker_pose is None:
                    marker_pose = self.marker_pose
                self.marker_pose = marker_pose
                # if marker_pose.position.y < 0.4:
                target_pose = make_pose(
                    np.array(
                        [
                            marker_pose.position.x,
                            marker_pose.position.y,
                            marker_pose.position.z + 0.25,
                        ]
                    ),
                    quat2mat(
                        np.array(
                            [
                                0.9122382281844055,
                                -0.40922287561112036,
                                0.007489168711762336,
                                -0.01737715360894841,
                            ]
                        )
                    ),
                )
                if not self.marker_found:
                    self.marker_found = True
                    new_marker_found = True

                self.missing_marker = False

            else:
                # if (self.marker_found or self.missing_marker) and self.marker_pose is not None:
                #     target_pose = make_pose(np.array([marker_pose.position.x,
                #                                       marker_pose.position.y,
                #                                       marker_pose.position.z+0.25]),
                #                             quat2mat(np.array([0.9122382281844055,
                #                                               -0.40922287561112036,
                #                                               0.007489168711762336,
                #                                               -0.01737715360894841
                #                             ])))
                #     self.marker_found = False
                #     self.missing_marker = True
                # else:
                goal = np.array([0.552220667184348, -0.25, 0.3502243150993263])
                goal_rot = quat2mat(
                    np.array(
                        [
                            0.9122382281844055,
                            -0.40922287561112036,
                            0.007489168711762336,
                            -0.01737715360894841,
                        ]
                    )
                )
                target_pose = make_pose(goal, goal_rot)
        else:
            goal = np.array([0.552220667184348, -0.15, 0.3502243150993263])
            goal_rot = quat2mat(
                np.array(
                    [
                        0.9122382281844055,
                        -0.40922287561112036,
                        0.007489168711762336,
                        -0.01737715360894841,
                    ]
                )
            )
            target_pose = make_pose(goal, goal_rot)
        return target_pose, new_marker_found

    def _get_obs(self):
        joint_states = self.panda_client.current_joints[:7]
        joint_states = np.concatenate([joint_states, np.array([0.04])])
        # ee_pose = self.panda_client.get_ee_pose(joint_states)
        # ee_pos = ee_pose[:3, 3]
        # obs = np.concatenate((joint_states, ee_pos))
        target_pose, marker_found = self.get_target_pose()
        goal = target_pose[:3, 3]
        goal_rot = target_pose[:3, :3]

        world_to_robot = np.eye(4)
        rtm = np.eye(4)
        rtm[:3, 3] = self._hp.robot_to_model
        pc = self.get_pc_obs()
        obs = AttrDict(
            robot_states=joint_states,
            goal=goal,
            goal_rot=goal_rot,
            camera_pose=world_to_robot,
            robot_to_model=rtm,
            model_to_robot=np.linalg.inv(rtm),
            ignore_coll=False,
            reset_geco=marker_found,
        )
        obs.update(pc)
        return obs


if __name__ == "__main__":
    env = PandaReachEnv()
    env.reset()
    # goal = env.goal
    # pose = Point(goal[0], goal[1], goal[2])
    # env.interactive_tool.make6DofMarker(InteractiveMarkerControl.MOVE_3D, pose, 'panda_link0', False)
    # env.interactive_tool.server.applyChanges()
    # env.create_interactive_marker(goal, 'goal')
    # env.menu_handler.insert('do stuff', callback=env.processFeedback)
