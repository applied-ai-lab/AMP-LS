import os
import numpy as np
import json
import rospy
import argparse
from visualization_msgs.msg import MarkerArray
from trajectory_msgs.msg import JointTrajectory
from core.rl.envs.gazebo_panda import GazeboPandaEnv
from core.utils.ros_utils import create_marker_message
from core.utils.general_utils import AttrDict

# use one of the following launch files for evaluation:
# roslaunch panda_moveit_config shapenet_empty_camera_connected.launch (no Panda in Gazebo)
# roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch (with Panda in Gazebo)
# roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch pipelin:=chomp (using CHOMP, with Panda in Gazebo)


def compute_path_length(
    env: GazeboPandaEnv, joint_trajectory: JointTrajectory
) -> float:
    prev_ee_pos = None
    distance = 0.0
    for point in joint_trajectory.points:
        # ee_pose_fk_response = env.robot_client.get_ee_pose(point.positions)
        # ee_pose = ee_pose_fk_response.pose_stamped[0].pose.position
        # ee_pos = np.array([ee_pose.x, ee_pose.y, ee_pose.z])

        ee_pose = env.robot_client.get_ee_pose(point.positions, use_kinematics=True)
        ee_pos = ee_pose[:3, 3]

        if prev_ee_pos is not None:
            distance += np.linalg.norm(prev_ee_pos - ee_pos)
        prev_ee_pos = ee_pos
    return distance


def evaluate(args):
    env = GazeboPandaEnv(
        AttrDict(
            arrange_mesh_scene=False,
            # robot_client_params=AttrDict(
            #     use_fake_controller=True
            # ),
        )
    )
    if args.planner_id != "" and args.planner_id != "CHOMP":
        env.robot_client.panda_arm.set_planner_id(args.planner_id)

    env.robot_client.panda_arm.set_planning_time(args.planning_time)

    # marker_pub = rospy.Publisher('rviz_visual_tools', MarkerArray, queue_size=1)

    with open(args.scene_config_path, "r") as fp:
        scene_config = json.loads(fp.read())

    num_traj = len(scene_config.keys())

    # prepare json output saving
    exp_dir = os.path.join(
        os.environ["EXP_DIR"], "motion_planner", args.planner_id, args.prefix
    )

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # load json file if exist
    filename = "ompl" + f"_{num_traj}_trajs"
    output_filename = filename + ".json"
    save_json_path = os.path.join(exp_dir, output_filename)
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as fp:
            output = json.load(fp)
            trajectories = output["trajectories"]
            planning_times = output["planning_times"]
            path_lengths = output["path_lengths"]
            normalised_path_lengths = output["normalised_path_lengths"]
            success_ids = output["success_ids"]
            i = output["i"]
        print(">>> Loaded input json: %s" % save_json_path)
    else:
        output = {}
        trajectories = {}
        planning_times = []
        path_lengths = []
        normalised_path_lengths = []
        success_ids = []
        i = 0

    rate = rospy.Rate(1.0)

    while i < num_traj:
        print(f"Planning Trajectory {i + 1}/{num_traj}...")

        # retrieve scene info from init config
        scene_info = scene_config["%03d" % i]["scene_info"]
        ee_jpos_xyz = np.array(scene_config["%03d" % i]["ee_jpos_xyz"])
        goal_jpos_xyz = np.array(scene_config["%03d" % i]["goal_jpos_xyz"])
        jpos = ee_jpos_xyz[:7]
        ee_x, ee_y, ee_z = ee_jpos_xyz[7:10]
        goal_jpos = goal_jpos_xyz[:7]
        g_x, g_y, g_z = goal_jpos_xyz[7:10]
        goal_quat = goal_jpos_xyz[10:]

        env.delete_model_client("table")
        env.clear_objects()
        # env.clear_octomap()

        print("Start position: ", jpos)
        env.robot_client.set_joint_state(jpos)
        rate.sleep()
        env.spawn_table()

        # set object positions
        env.scene.remove_world_object()
        # env.clear_octomap()
        env.arrange_objects_to(scene_info)
        env.add_box2scene(
            "table", [0.6, 0, -0.2], orientation=[0, 0, 0, 1], size=[1.0, 1.6, 0.2]
        )

        env.clear_octomap()

        # rate.sleep()
        rospy.sleep(2.0)
        print("Ready to plan")

        if args.planner_id == "CHOMP":  # Only joint-space goals are supported in CHOMP.
            (
                success,
                plan_msg,
                planning_time,
                error_code,
            ) = env.robot_client.plan_ee_to_joint_state(goal_jpos)
        elif args.ori:
            # success, plan_msg, planning_time, error_code = env.robot_client.plan_ee_to([g_x, g_y, g_z], goal_quat, orientation_tolerance=np.deg2rad(10.)) # 10 degrees
            success, plan_msg, planning_time, error_code = env.robot_client.plan_ee_to(
                [g_x, g_y, g_z], goal_quat
            )  # 10 degrees
        else:
            success, plan_msg, planning_time, error_code = env.robot_client.plan_ee_to(
                [g_x, g_y, g_z], orientation_tolerance=2 * np.pi
            )
        print("Planned a path in {} seconds".format(planning_time))

        # save planned path and scene info to trajectories output
        trajectories["%03d" % i] = {}

        trajectories["%03d" % i]["ee_jpos"] = [(jpos / np.pi * 180.0).tolist()]
        trajectories["%03d" % i]["pred_ee_xyz"] = [[0.0, 0.0, 0.0]]
        trajectories["%03d" % i]["target_xyz"] = [g_x, g_y, g_z]
        trajectories["%03d" % i]["scene_info"] = scene_info

        if success:
            print("success")
            planning_times.append(planning_time)

            for point in plan_msg.joint_trajectory.points:
                trajectories["%03d" % i]["ee_jpos"].append(
                    (np.array(point.positions) / np.pi * 180.0).tolist()
                )
                trajectories["%03d" % i]["pred_ee_xyz"].append([0.0, 0.0, 0.0])

            print("Computing path length...")
            path_length = compute_path_length(env, plan_msg.joint_trajectory)
            path_lengths.append(path_length)

            print("Normalising path length...")
            dist_init_ee_to_goal = np.linalg.norm(
                ee_jpos_xyz[7:10] - goal_jpos_xyz[7:10]
            )
            normalised_path_length = path_length / dist_init_ee_to_goal
            normalised_path_lengths.append(normalised_path_length)

            success_ids.append(i)

        i += 1

        print("Updating output...")
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

        # rate.sleep()

    # prepare evaluation metrics: planning time, path length, normalised path length, success rate
    mean_time = np.mean(planning_times)
    std_time = np.std(planning_times)
    mean_path_length = np.mean(path_lengths)
    std_path_length = np.std(path_lengths)
    mean_normalised_path_length = np.mean(normalised_path_lengths)
    std_normalised_path_length = np.std(normalised_path_lengths)
    success_rate = len(success_ids) / float(num_traj)

    result = f"Planning time: {mean_time:.4f} +- {std_time:.4f}\n"
    result += f"Path length: {mean_path_length:.4f} +- {std_path_length:.4f}\n"
    result += f"Normalised path length: {mean_normalised_path_length:.4f} +- {std_normalised_path_length:.4f}\n"
    result += f"Success rate: {success_rate * 100:.1f}\n\n"
    result += "Planning time: " + ", ".join(str(e) for e in planning_times) + "\n"
    result += "Path length: " + ", ".join(str(e) for e in path_lengths) + "\n"
    result += (
        "Normalised path length: "
        + ", ".join(str(e) for e in normalised_path_lengths)
        + "\n"
    )
    result += "Success id: " + ", ".join(str(e) for e in success_ids)

    print(result)

    # save planned paths in a json file
    with open(save_json_path, "w") as fp:
        json.dump(output, fp, indent=2, sort_keys=True)
    print(">>> Saved output json: %s" % save_json_path)

    # save evaluation metrics in a txt file
    output_txt = filename + ".txt"
    save_text_path = os.path.join(exp_dir, output_txt)
    with open(save_text_path, "w") as f:
        f.write(result)
        f.close()


if __name__ == "__main__":
    cmdl_parser = argparse.ArgumentParser(
        description="Generate obstacle collision data."
    )

    cmdl_parser.add_argument("--save_every_traj", action="store_true")
    cmdl_parser.add_argument(
        "--prefix", default=None, help="Dir to save trajectory json file."
    )
    cmdl_parser.add_argument(
        "--scene_config_path",
        default=os.path.join(
            os.environ["DATA_DIR"], "gazebo/env_configs/scene_config_010_trajs.json"
        ),
        help="init config json file.",
    )
    cmdl_parser.add_argument(
        "--planner_id",
        default="RRTConnectkConfigDefault",
        help="OMPL planner id or CHOMP. \
        Currently used: RRTConnectkConfigDefault, RRTstarkConfigDefault, LazyPRMstarkConfigDefault, LBKPIECEkConfigDefault. \
        See https://github.com/ascane/panda_moveit_config/blob/melodic-devel/config/ompl_planning.yaml for more available planners. \
        If CHOMP, please use the launch file with the chomp pipeline. \
        Planning time needs to be set in panda_moveit_config/config/chomp_planning.yaml.",
    )
    cmdl_parser.add_argument("--planning_time", type=float, default=1.0)
    cmdl_parser.add_argument(
        "--ori", action="store_true", help="Use orientation constraint (10 degrees)."
    )
    args = cmdl_parser.parse_args()

    assert args.prefix is not None, "Prefix is None."
    evaluate(args)
