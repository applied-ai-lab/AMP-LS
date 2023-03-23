#!/usr/bin/env python

import os, sys
import copy
import numpy as np
import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState
import moveit_commander
from urdfpy import URDF
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from core.utils.general_utils import AttrDict, ParamDict
from core.utils.transform_utils import mat2quat

import franka_gripper
import franka_gripper.msg


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if isinstance(goal, list):
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif isinstance(goal, PoseStamped):
        return all_close(goal.pose, actual.pose, tolerance)

    elif isinstance(goal, Pose):
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class PandaClient(object):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

        self.ee_safety_zone = np.array(self._hp.ee_safety_zone)
        self.kinematics = URDF.load(self._hp.robot_urdf)

        low_joint_limits, high_joint_limits = self.kinematics.joint_limit_cfgs
        self.low_joint_vals = np.fromiter(low_joint_limits.values(), dtype=float)
        self.high_joint_vals = np.fromiter(high_joint_limits.values(), dtype=float)
        self.low_joint_vals[-1] = 0.04

        # connect to state validity service client
        rospy.wait_for_service("/check_state_validity")
        self.moveit_sv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")

        # Outer level interface to the robot:
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        panda_arm = moveit_commander.MoveGroupCommander(self._hp.group_name)
        panda_arm.set_pose_reference_frame("panda_link0")
        panda_arm.allow_replanning(self._hp.allow_replanning)
        panda_arm.set_planning_time(self._hp.planning_time)

        self.panda_arm = panda_arm

        # We can get the name of the reference frame for this robot:
        print(
            "============ Reference frame: ",
            self.panda_arm.get_planning_frame(),
            self.robot.get_planning_frame(),
        )

        eef_link = self.panda_arm.get_end_effector_link()
        self.ee_link = eef_link
        print("============ End effector: %s" % eef_link)

        group_names = self.robot.get_group_names()
        self.group_names = group_names
        print("============ Robot Groups:", self.robot.get_group_names())

        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

        if self._hp.load_gripper:
            gripper_homing_client = actionlib.SimpleActionClient(
                "/franka_gripper/homing", franka_gripper.msg.HomingAction
            )
            gripper_move_client = actionlib.SimpleActionClient(
                "/franka_gripper/move", franka_gripper.msg.MoveAction
            )
            gripper_grasp_client = actionlib.SimpleActionClient(
                "/franka_gripper/grasp", franka_gripper.msg.GraspAction
            )

            print("Waiting for gripper homing server")
            gripper_homing_client.wait_for_server()
            #
            print("Waiting for gripper move server")
            gripper_move_client.wait_for_server()
            #
            print("Waiting for gripper grasp server")
            gripper_grasp_client.wait_for_server()

        self.cmd_joint_pos_pub = rospy.Publisher(
            "franka_motion_control/joint_command", JointState, queue_size=1
        )
        self.joint_states_sub = rospy.Subscriber(
            "franka_motion_control/joint_states", JointState, self._update_joint_states
        )
        rospy.loginfo("Waiting for /franka_motion_control/joint_states")
        rospy.wait_for_message("/franka_motion_control/joint_states", JointState)
        rospy.loginfo("Connected to Franka Joint States")

        if self._hp.load_gripper:
            self.gripper_homing_client = gripper_homing_client
            self.gripper_move_client = gripper_move_client
            self.gripper_grasp_client = gripper_grasp_client

        self.home_pose_joint_values = self._hp.home_pose_joint_values

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "group_name": "panda_arm",
                "load_gripper": True,
                "ee_safety_zone": [[0.30, 0.76], [-0.18, 0.18], [0.01, 0.4]],
                "moveit_safety_zone": [
                    [-np.inf, np.inf],
                    [-np.inf, np.inf],
                    [-np.inf, np.inf],
                ],
                "home_pose_joint_values": [
                    0,
                    -numpy.pi / 4.0,
                    0,
                    -0.75 * numpy.pi,
                    0,
                    numpy.pi / 2.0,
                    numpy.pi / 4.0,
                ],
                "robot_urdf": "./core/data/scene_collision/assets/data/panda/panda.urdf",
                "load_moveit": False,
                "goal_position_tolerance": 0.02,
                "goal_joint_tolerance": 0.01,
                "allow_replanning": False,
                "planning_time": 1.0,
            }
        )
        return default_dict

    def get_ee_pose(self, joints):
        # joint_state_msg = form_joint_state_msg(joints)
        #
        # # prepare a skeleton request message
        # req = GetPositionFKRequest()
        # req.header.frame_id = 'panda_link0'
        # req.fk_link_names = ['panda_link8']
        #
        # # add joint state to request
        # req.robot_state.joint_state = joint_state_msg
        #
        # # fire the service request to compute_fk
        # resp = self.moveit_fk(req)
        # return resp
        if len(joints) == 7:
            joints = np.concatenate([joints, np.array([0.04])])
        pose = self.kinematics.link_fk(joints, "panda_link8")
        return pose

    def _update_state(self, msg):
        self.O_P_EE_prev = self.O_P_EE
        self.O_V_EE_prev = self.O_V_EE
        self.O_P_EE_timestamp_secs_prev = self.O_P_EE_timestamp_secs
        self.O_F_EE_prev = self.O_F_EE
        self.K_F_EE_prev = self.K_F_EE
        self.O_O_EE_prev = self.O_O_EE
        self.O_T_EE_prev = self.O_T_EE

        self.O_P_EE = np.array(msg.O_T_EE[-4:-1])
        self.O_O_EE = np.array(msg.O_T_EE[:-4])
        self.O_T_EE = np.array(msg.O_T_EE).reshape((4, 4)).T

        self.O_P_EE_timestamp_secs = msg.header.stamp.secs + msg.header.stamp.nsecs * (
            1e-9
        )

        self.O_F_EE = np.array(msg.O_F_ext_hat_K)
        self.K_F_EE = np.array(msg.K_F_ext_hat_K)

        if self.O_P_EE_prev is not None:
            dt = self.O_P_EE_timestamp_secs - self.O_P_EE_timestamp_secs_prev
            assert dt > 0
            self.O_V_EE = (self.O_P_EE - self.O_P_EE_prev) / dt

        # Filter velocity (not sure if this is necessary)
        if self.O_V_EE_prev is not None:
            self.O_V_EE = 0.5 * self.O_V_EE + 0.5 * self.O_V_EE_prev

        self.franka_state = msg
        # print(self.get_ee_state())

    def _update_joint_states(self, msg):
        self.joint_pos = np.array(msg.position)
        self.joint_vel = np.array(msg.velocity)

    def ee_inside_safety_zone(self, xyz):
        return (
            xyz[0] >= self.ee_safety_zone[0][0]
            and xyz[0] <= self.ee_safety_zone[0][1]
            and xyz[1] >= self.ee_safety_zone[1][0]
            and xyz[1] <= self.ee_safety_zone[1][1]
            and xyz[2] >= self.ee_safety_zone[2][0]
            and xyz[2] <= self.ee_safety_zone[2][1]
        )

    def plan_ee_to(self, pose_goal, clip=False):
        if clip:
            pose_goal = np.clip(
                pose_goal, self.ee_safety_zone[:, 0], self.ee_safety_zone[:, 1]
            )
        if not self.ee_inside_safety_zone(pose_goal):
            raise Exception(
                "Goal ee pose should be inside the safety zone {}".format(
                    pose_goal, self.ee_safety_zone
                )
            )

        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        current_pose = self.panda_arm.get_current_pose()
        pose_target = copy.deepcopy(current_pose)
        pose_target.pose.position.x = pose_goal[0]
        pose_target.pose.position.y = pose_goal[1]
        pose_target.pose.position.z = pose_goal[2]

        self.panda_arm.set_pose_target(pose_target)

        # Now, we call the planner to compute the plan and execute it.
        plan = self.panda_arm.plan()
        self.panda_arm.clear_pose_targets()
        return plan

    def plan_joints_to(self, joints, wait_to_finish=True):
        # Now, we call the planner to compute the plan and execute it.
        self.panda_arm.clear_pose_targets()
        plan = self.panda_arm.go(joints=joints, wait=wait_to_finish)
        self.panda_arm.stop()
        return plan

    def move_ee_to(
        self,
        pose_goal,
        orientation=None,
        clip=False,
        wait_to_finish=True,
        velocity=None,
    ):
        pre_controller = None
        if not self.controller_is_running("position_joint_trajectory_controller"):
            pre_controller = self.get_current_controller()
            print("Switching to position control")
            self.switch_controllers("position_joint_trajectory_controller")

        if clip:
            pose_goal = np.clip(
                pose_goal, self.ee_safety_zone[:, 0], self.ee_safety_zone[:, 1]
            )
        if not self.ee_inside_safety_zone(pose_goal):
            raise Exception(
                "Goal ee pose should be inside the safety zone {}".format(
                    pose_goal, self.ee_safety_zone
                )
            )

        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        current_pose = self.panda_arm.get_current_pose()
        pose_target = copy.deepcopy(current_pose)
        pose_target.pose.position.x = pose_goal[0]
        pose_target.pose.position.y = pose_goal[1]
        pose_target.pose.position.z = pose_goal[2]

        if orientation is not None:
            pose_target.pose.orientation.x = orientation[0]
            pose_target.pose.orientation.y = orientation[1]
            pose_target.pose.orientation.z = orientation[2]
            pose_target.pose.orientation.w = orientation[3]

        self.panda_arm.set_pose_target(pose_target)

        res = self.panda_arm.plan()
        plan = []
        for point in res[1].joint_trajectory.points:
            plan.append(point.positions)

        # Now, we call the planner to compute the plan and execute it.
        self.panda_arm.go(wait=wait_to_finish)
        # Calling `stop()` ensures that there is no residual movement
        self.panda_arm.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.panda_arm.clear_pose_targets()

        current_pose = self.panda_arm.get_current_pose()

        if pre_controller is not None:
            self.switch_controllers(pre_controller)

        return all_close(pose_target, current_pose, 0.01), plan

    def move_joints_to(self, joints, wait_to_finish=True):
        # Now, we call the planner to compute the plan and execute it.
        plan = self.panda_arm.go(joints=joints, wait=wait_to_finish)
        if not plan:
            while not plan:
                ans = input("Did you reconfigure the panda arm?[Yes/No]")
                if ans.lower() == "yes" or ans.lower() == "y":
                    self.recover_from_errors()
                    self.panda_arm.clear_pose_targets()
                    plan = self.panda_arm.go(joints=joints, wait=wait_to_finish)
                    if self.is_in_collision_mode():
                        self.recover_from_errors()
        # Calling `stop()` ensures that there is no residual movement
        self.panda_arm.stop()
        return True

    def set_joint_pos(self, joint_states, joint_velocities=None):
        goal_state = JointState()
        goal_state.position = joint_states

        robot_state = RobotState()
        robot_state.joint_state = goal_state
        # if not self.has_collision(robot_state):
        if joint_velocities is None:
            joint_velocities = np.zeros(7)
        goal_state.velocity = joint_velocities

        self.cmd_joint_pos_pub.publish(goal_state)

    def recover_from_errors(self):
        # goal = ErrorRecoveryActionGoal()
        goal = ErrorRecoveryGoal()
        self.error_recovery_client.send_goal(goal)
        print("Waiting for recovery goal")
        self.error_recovery_client.wait_for_result()
        print("Done")
        return self.error_recovery_client.get_result()

    def is_in_contact_mode(self):
        if self.franka_state is None:
            return False

        return any(self.franka_state.joint_contact)

    def is_in_collision_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_REFLEX

    def is_in_move_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_MOVE

    def is_in_idle_mode(self):
        if self.franka_state is None:
            return False

        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_IDLE

    def is_in_user_stop_mode(self):
        if self.franka_state is None:
            return False
        return self.franka_state.robot_mode == FrankaState.ROBOT_MODE_USER_STOPPED

    def has_collision(self, robot_state: RobotState):
        """
        Check for self collision or collision of the panda arm with other collision objects.
        """
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = "panda_arm"
        result = self.moveit_sv.call(gsvr)
        return not result.valid

    def move_gripper_to(self, width, speed=0.05):
        goal = franka_gripper.msg.MoveGoal(width=width, speed=speed)
        self.gripper_move_client.send_goal(goal)
        self.gripper_move_client.wait_for_result()
        return self.gripper_move_client.get_result()

    def gripper_homing(self):
        goal = franka_gripper.msg.HomingGoal()
        self.gripper_homing_client.send_goal(goal)
        self.gripper_homing_client.wait_for_result()
        return self.gripper_homing_client.get_result()

    def gripper_grasp(self, width, speed=0.05, force=10):
        epsilon = franka_gripper.msg.GraspEpsilon(inner=0.01, outer=0.01)
        goal = franka_gripper.msg.GraspGoal(
            width=width, speed=speed, epsilon=epsilon, force=force
        )

        self.gripper_grasp_client.send_goal(goal)
        self.gripper_grasp_client.wait_for_result()
        return self.gripper_grasp_client.get_result()

    def moveit_move_gripper_to(self, pose_goal):
        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        self.hand.set_pose_target(pose_goal)

        # Now, we call the planner to compute the plan and execute it.
        plan = self.hand.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.hand.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.hand.clear_pose_targets()

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.hand.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    @property
    def current_joints(self):
        return self.joint_pos

    def add_box2scene(self, name, position, orientation, size):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.panda_arm.get_planning_frame()
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        pose_stamped.pose.orientation.x = orientation[0]
        pose_stamped.pose.orientation.y = orientation[1]
        pose_stamped.pose.orientation.z = orientation[2]
        pose_stamped.pose.orientation.w = orientation[3]
        self.scene.add_box(name, pose_stamped, size)


if __name__ == "__main__":
    rospy.init_node("test")
    config = AttrDict()
    panda = PandaClient(config)

    current_joint_pos = panda.joint_pos
    print(mat2quat(panda.get_ee_pose(current_joint_pos)))
    goal_joint_pos = copy.deepcopy(current_joint_pos)

    traj = []
    for i in range(10):
        goal_joint_pos[2] += 0.003
        traj.append(copy.deepcopy(goal_joint_pos))

    for i in range(10):
        goal_joint_pos[2] -= 0.004
        traj.append(copy.deepcopy(goal_joint_pos))

    for i in range(10):
        goal_joint_pos[2] -= 0.006
        traj.append(copy.deepcopy(goal_joint_pos))

    for i in range(40):
        goal_joint_pos[2] += 0.004
        goal_joint_pos[3] += 0.003
        traj.append(copy.deepcopy(goal_joint_pos))

    # goal_joint_pos[2] += 0.01

    # rate = rospy.Rate(100)
    # while not rospy.is_shutdown():
    #     panda.set_joint_pos(goal_joint_pos)
    #     rate.sleep()

    rate = rospy.Rate(100)
    velocity = []
    for goal_joint_pos in traj:
        print(goal_joint_pos)
        panda.set_joint_pos(goal_joint_pos)
        velocity.append(panda.joint_vel[2])
        rate.sleep()

    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(velocity)), velocity)
    plt.show()
    # while not rospy.is_shutdown():
    #     panda.set_joint_pos(goal_joint_pos)
    #     rate.sleep()
    #     print('hi')
