import os
import json
import numpy as np
import argparse

from core.planner.envs.gazebo_panda import GazeboPandaEnv
from core.utils.general_utils import AttrDict
import rospy

# use one of the following launch files for scene config generation:
# roslaunch panda_moveit_config shapenet_empty_camera_connected.launch (no Panda in Gazebo)
# roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch (with Panda in Gazebo)


def check_scene_not_too_close(scene_info, goal_xyz, thresh=0.1):
    for obj_input in scene_info:
        pose = np.array(obj_input["pose"])  # convert list back to ndarray
        pos = pose[:3, 3]
        if np.linalg.norm(pos - goal_xyz) < thresh:
            return False
    return True


def generate_init_config(args):
    bounds = np.array([[0.3, -0.7, 0.0], [0.8, 0.7, 0.3]])
    start_bounds = np.array([[0.2, -0.8, 0.0], [0.8, 0.8, 0.3]])
    env_params = AttrDict(
        robot_client_params=AttrDict(use_fake_controller=True),
        arrange_mesh_scene=False if args.scene_config_path is not None else True,
    )
    env = GazeboPandaEnv(env_params)

    # prepare json output saving
    assert args.save_dir != "", "save_dir is empty"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output = {}

    if args.save_pc:
        filename = "scene_config_%03d_trajs_w_pc" % args.n_scenes
    else:
        filename = "scene_config_%03d_trajs" % args.n_scenes
    filename += args.suffix

    if args.scene_config_path is not None:
        with open(args.scene_config_path, "r") as fp:
            scene_config = json.loads(fp.read())

    for i in range(args.n_scenes):
        print(f"Generating Scene Config {i + 1}/{args.n_scenes}...")

        if args.scene_config_path is not None:
            scene_info = scene_config["%03d" % i]["scene_info"]
            start_states = scene_config["%03d" % i]["ee_jpos_xyz"]
            goal_states = scene_config["%03d" % i]["goal_jpos_xyz"]
            env.arrange_objects_to(scene_info)
        else:
            scene_info = env.arrange_objects()
            start_states = env.sample_valid_jpos_xyz(start_bounds).tolist()
            valid = False
            while not valid:
                goal_states = env.sample_valid_jpos_xyz(bounds).tolist()
                goal_xyz = np.array(goal_states[7:10])
                if np.linalg.norm(
                    np.array(start_states[7:10]) - goal_xyz
                ) > 0.5 and check_scene_not_too_close(scene_info, goal_xyz, thresh=0.3):
                    valid = True

        output["%03d" % i] = {}
        output["%03d" % i]["scene_info"] = scene_info
        output["%03d" % i]["ee_jpos_xyz"] = start_states
        output["%03d" % i]["goal_jpos_xyz"] = goal_states

        # save point cloud obs
        if args.save_pc:
            env.clear_octomap()
            rospy.sleep(2.0)
            obs = env.get_obs_as_list(output["%03d" % i])
            output["%03d" % i]["obs"] = obs

    # print(output)

    output_filename = filename + ".json"
    save_json_path = os.path.join(args.save_dir, output_filename)
    with open(save_json_path, "w") as fp:
        json.dump(output, fp, indent=2, sort_keys=True)
    print(">>> Saved output json: %s" % save_json_path)


if __name__ == "__main__":
    cmdl_parser = argparse.ArgumentParser(
        description="Generate obstacle collision data."
    )
    cmdl_parser.add_argument(
        "--save_dir",
        default=os.path.join(os.environ["DATA_DIR"], "gazebo/env_configs"),
        help="Dir to save json file.",
    )
    cmdl_parser.add_argument(
        "--n_scenes",
        type=int,
        default=10,
        help="Number of trajectories in the init config.",
    )
    cmdl_parser.add_argument("--suffix", type=str, default="")
    cmdl_parser.add_argument("--save_pc", action="store_true")
    cmdl_parser.add_argument("--scene_config_path", type=str, default=None)

    args = cmdl_parser.parse_args()

    generate_init_config(args)
