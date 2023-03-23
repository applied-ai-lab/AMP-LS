import os, sys
import copy
import numpy as np
import rospy
from typing import List, Sequence

from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from geometry_msgs.msg import Pose, Vector3, Point, PoseStamped
from std_msgs.msg import ColorRGBA
import tf.transformations
from core.utils.transform_utils import mat2quat
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from tf.broadcaster import TransformBroadcaster


def form_joint_state_msg(jpos) -> JointState:
    # prepare a skeleton message to hold the query joint configuration
    joint_state_msg = JointState()
    joint_state_msg.name = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    joint_state_msg.velocity = [0.0] * 7
    joint_state_msg.effort = [0.0] * 7

    # complete JointState message
    joint_state_msg.header.stamp = rospy.Time.now()
    joint_state_msg.position = jpos

    return joint_state_msg


def form_pose_from_ndarray(obj_pose):
    pose = Pose()
    pose.position.x = obj_pose[0, 3]
    pose.position.y = obj_pose[1, 3]
    pose.position.z = obj_pose[2, 3]
    quat = mat2quat(obj_pose[:3, :3])
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]

    return pose


def form_joint_trajectory_msg(jpos) -> JointTrajectory:
    joint_traj_msg = JointTrajectory()
    joint_traj_msg.joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    point = JointTrajectoryPoint()
    point.positions = jpos
    point.time_from_start = rospy.Duration(0.1)
    joint_traj_msg.points = [point]

    return joint_traj_msg


def tf_matrix(transform):
    trans = tf.transformations.translation_matrix(
        (transform.translation.x, transform.translation.y, transform.translation.z)
    )

    rot = tf.transformations.quaternion_matrix(
        (
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )
    )
    combined = np.matmul(trans, rot)

    return combined


def create_marker_message(
    id: int, x: float, y: float, z: float, rgba: Sequence
) -> Marker:
    # generic
    marker_msg = Marker()
    marker_msg.header.stamp = rospy.get_rostime()
    marker_msg.header.frame_id = "panda_link0"
    marker_msg.ns = ""
    marker_msg.id = id
    marker_msg.type = 2  # sphere
    marker_msg.action = 0
    marker_msg.pose.orientation = [0] * 4
    marker_msg.scale = Vector3(0.05, 0.05, 0.05)
    marker_msg.lifetime = rospy.Duration(0)

    # specific
    marker_msg.color = ColorRGBA(*rgba)
    marker_msg.pose = Pose()
    marker_msg.pose.position.x = x
    marker_msg.pose.position.y = y
    marker_msg.pose.position.z = z

    return marker_msg


class InteractiveMarkerTool(object):
    def __init__(self):
        self.server = InteractiveMarkerServer("interactive_marker")
        self.menu_handler = MenuHandler()
        self.current_pos = None
        self.current_quat = None

    def processFeedback(self, feedback):
        # s = "Feedback from marker '" + feedback.marker_name
        # s += "' / control '" + feedback.control_name + "'"
        self.feedback = feedback
        self.current_pos = (
            self.feedback.pose.position.x,
            self.feedback.pose.position.y,
            self.feedback.pose.position.z,
        )
        self.current_quat = (
            self.feedback.pose.orientation.x,
            self.feedback.pose.orientation.y,
            self.feedback.pose.orientation.z,
            self.feedback.pose.orientation.w,
        )
        # print(feedback)
        # if feedback.mouse_point_valid:
        #     self.current_pos = (self.feedback.mouse_point.x,
        #                         self.feedback.mouse_point.y,
        #                         self.feedback.mouse_point.z)
        # mp = ""
        # if feedback.mouse_point_valid:
        #     mp = " at " + str(feedback.mouse_point.x)
        #     mp += ", " + str(feedback.mouse_point.y)
        #     mp += ", " + str(feedback.mouse_point.z)
        #     mp += " in frame " + feedback.header.frame_id
        # print(mp)
        # if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
        #     rospy.loginfo( s + ": button click" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
        #     rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        #     rospy.loginfo( s + ": pose changed")
        # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
        #     rospy.loginfo( s + ": mouse down" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
        #     rospy.loginfo( s + ": mouse up" + mp + "." )
        self.server.applyChanges()

    def make6DofMarker(
        self,
        interaction_mode,
        position,
        frame_id,
        show_6dof=False,
        scale=0.1,
        orientation=None,
    ):
        self.current_pos = (position.x, position.y, position.z)
        if orientation is not None:
            self.current_quat = (
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        else:
            self.current_quat = (0, 0, 0, 1)
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.pose.position = position
        int_marker.pose.orientation = orientation
        int_marker.scale = scale

        int_marker.name = "simple_6dof"
        int_marker.description = "Simple 6-DOF Control"

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = {
                InteractiveMarkerControl.MOVE_3D: "MOVE_3D",
                InteractiveMarkerControl.ROTATE_3D: "ROTATE_3D",
                InteractiveMarkerControl.MOVE_ROTATE_3D: "MOVE_ROTATE_3D",
            }
            int_marker.name += "_" + control_modes_dict[interaction_mode]
            int_marker.description = "3D Control"
            if show_6dof:
                int_marker.description += " + 6-DOF controls"
            int_marker.description += "\n" + control_modes_dict[interaction_mode]

        if show_6dof:
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            # normalizeQuaternion(control.orientation)
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            # normalizeQuaternion(control.orientation)
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            # normalizeQuaternion(control.orientation)
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            # normalizeQuaternion(control.orientation)
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            # normalizeQuaternion(control.orientation)
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            # normalizeQuaternion(control.orientation)
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def makeViewFacingMarker(self, position, frame_id, scale=0.1):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.pose.position = position
        int_marker.scale = scale

        int_marker.name = "view_facing"
        int_marker.description = "View Facing 6-DOF"

        # make a control that rotates around the view axis
        control = InteractiveMarkerControl()
        control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        control.orientation.w = 1
        control.name = "rotate"
        int_marker.controls.append(control)

        # create a box in the center which should not be view facing,
        # but move in the camera plane.
        control = InteractiveMarkerControl()
        control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.independent_marker_orientation = True
        control.name = "move"
        control.markers.append(self.makeBox(int_marker))
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)

    def makeMovingMarker(self, position, scale=0.1):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = frame_id
        int_marker.pose.position = position
        int_marker.scale = scale

        int_marker.name = "moving"
        int_marker.description = "Marker Attached to a\nMoving Frame"

        control = InteractiveMarkerControl()
        control.orientation.w = 1
        control.orientation.x = 1
        control.orientation.y = 0
        control.orientation.z = 0
        normalizeQuaternion(control.orientation)
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(copy.deepcopy(control))

        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.always_visible = True
        control.markers.append(self.makeBox(int_marker))
        int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)

    def makeBox(self, msg):
        marker = Marker()

        # marker.type = Marker.CUBE
        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale * 0.45
        marker.scale.y = msg.scale * 0.45
        marker.scale.z = msg.scale * 0.45
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0
        marker.color.a = 1.0

        return marker

    def makeBoxControl(self, msg):
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeBox(msg))
        msg.controls.append(control)
        return control

    @property
    def marker_pos(self):
        return self.current_pos

    @property
    def marker_quat(self):
        return self.current_quat
