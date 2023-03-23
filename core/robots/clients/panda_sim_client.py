#! /usr/bin/env python

import sys
import copy

import numpy as np
from numpy.random import default_rng
import rospy
from typing import List
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import *
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
import moveit_commander
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from urdfpy import URDF

from core.utils.general_utils import ParamDict, AttrDict
from core.utils.ros_utils import form_joint_state_msg, form_joint_trajectory_msg


class PandaSimClient(object):
    def __init__(self, config):
        # set parameters
        self._hp = self._default_hparams().overwrite(config)

        # connect to forward kinematics service client
        rospy.wait_for_service("/compute_fk")
        print("Found compute_fk service...")
        try:
            self.moveit_fk = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        except rospy.ServiceException as e:
            rospy.logerror(f"Service call failed: {e}")
        print("... and connected.")

        # init moveit commander interface
        moveit_commander.roscpp_initialize(sys.argv)
        # instantiate a robot commaner object
        self.robot = moveit_commander.RobotCommander()
        self.kinematics = URDF.load(self._hp.robot_urdf)

        # instantiate a planning scene interface
        self.panda_arm = moveit_commander.MoveGroupCommander(self._hp.group_name)
        self.panda_arm.set_planner_id(self._hp.planner_id)

        # Slow down the max velocity and acceleration of the arm
        self.panda_arm.set_max_velocity_scaling_factor(1.0)
        self.panda_arm.set_max_acceleration_scaling_factor(1.0)

        self.panda_arm.allow_replanning(False)
        self.panda_arm.set_planning_time(1.0)

        # joint state or trajectory publisher to control the robot
        if self._hp.use_fake_controller:
            self.js_pub = self.js_pub = rospy.Publisher(
                "/move_group/fake_controller_joint_states", JointState, queue_size=1
            )
        else:
            self.js_pub = rospy.Publisher(
                "/position_joint_trajectory_controller/command",
                JointTrajectory,
                queue_size=1,
            )

    def get_current_state(self):
        current_joint_values = self.panda_arm.get_current_joint_values()
        joint_state_msg = form_joint_state_msg(current_joint_values)

        robot_state = RobotState()
        robot_state.joint_state = joint_state_msg

        return robot_state

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "group_name": "panda_arm",  # group name for MoveGroupCommander
                "ee_name": "panda_link8",  # 'panda_rightfinger'
                "robot_urdf": "./core/data/scene_collision/assets/data/panda/panda.urdf",
                "planner_id": "RRTConnectkConfigDefault",  # OMPL planner id
                "use_fake_controller": False,
            }
        )
        return default_dict

    def set_joint_state(self, jpos):
        if self._hp.use_fake_controller:
            joint_msg = form_joint_state_msg(jpos)
        else:
            joint_msg = form_joint_trajectory_msg(jpos)

        for _ in range(3):
            self.js_pub.publish(joint_msg)

    def get_ee_pose(self, joints, use_kinematics=False):
        if use_kinematics:
            if len(joints) == 7:
                joints = np.concatenate([joints, np.array([0.04])])
            pose = self.kinematics.link_fk(joints, self._hp.ee_name)
            return pose

        joint_state_msg = form_joint_state_msg(joints)

        # prepare a skeleton request message
        req = GetPositionFKRequest()
        req.header.frame_id = "panda_link0"
        req.fk_link_names = ["panda_link8"]

        # add joint state to request
        req.robot_state.joint_state = joint_state_msg

        # fire the service request to compute_fk
        resp = self.moveit_fk(req)
        return resp

    def plan_ee_to(self, pose_goal, pose_orientation=None, orientation_tolerance=None):
        current_pose = self.panda_arm.get_current_pose()
        pose_target = copy.deepcopy(current_pose)
        pose_target.pose.orientation.w = 1.0
        if pose_orientation is not None:
            pose_target.pose.orientation.x = pose_orientation[0]
            pose_target.pose.orientation.y = pose_orientation[1]
            pose_target.pose.orientation.z = pose_orientation[2]
            pose_target.pose.orientation.w = pose_orientation[3]
        pose_target.pose.position.x = pose_goal[0]
        pose_target.pose.position.y = pose_goal[1]
        pose_target.pose.position.z = pose_goal[2]

        self.panda_arm.set_pose_target(pose_target)
        if orientation_tolerance is not None:
            self.panda_arm.set_goal_orientation_tolerance(orientation_tolerance)

        plan = self.panda_arm.plan()
        self.panda_arm.clear_pose_targets()
        return plan

    def plan_ee_to_joint_state(self, joint_goal, orientation_tolerance=None):
        self.panda_arm.set_joint_value_target(form_joint_state_msg(joint_goal))
        if orientation_tolerance is not None:
            self.panda_arm.set_goal_orientation_tolerance(orientation_tolerance)

        plan = self.panda_arm.plan()
        self.panda_arm.clear_pose_targets()
        return plan

    def move_ee_to(self, pose_goal, pose_orientation=None, orientation_tolerance=None):
        current_pose = self.panda_arm.get_current_pose()
        pose_target = copy.deepcopy(current_pose)
        pose_target.pose.orientation.w = 1.0
        if pose_orientation is not None:
            pose_target.pose.orientation.x = pose_orientation[0]
            pose_target.pose.orientation.y = pose_orientation[1]
            pose_target.pose.orientation.z = pose_orientation[2]
            pose_target.pose.orientation.w = pose_orientation[3]
        pose_target.pose.position.x = pose_goal[0]
        pose_target.pose.position.y = pose_goal[1]
        pose_target.pose.position.z = pose_goal[2]

        self.panda_arm.set_pose_target(pose_target)
        if orientation_tolerance is not None:
            self.panda_arm.set_goal_orientation_tolerance(orientation_tolerance)

        plan = self.panda_arm.plan()
        self.panda_arm.go()
        self.panda_arm.clear_pose_targets()
        return plan

    def plan_joints_to(self, joints, wait_to_finish=True):
        # Now, we call the planner to compute the plan and execute it.
        self.panda_arm.clear_pose_targets()
        plan = self.panda_arm.go(joints=joints, wait=wait_to_finish)
        self.panda_arm.stop()
        return plan
