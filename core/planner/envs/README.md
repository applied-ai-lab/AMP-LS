# Moveit planning

## Set up ROS + Moveit + Gazebo

Clone the forked `panda_moveit_config`.

(In `catkin_ws/src`)
```
git clone https://github.com/ascane/panda_moveit_config.git
git checkout lsmp_gazebo
```

```
git clone https://github.com/junjungoal/franka_ros.git
git checkout lsmp
```

## Set up Shapenet dataset

To evaluate a model in diverse scenes, download the ShapeNetSem meshes [here](https://shapenet.org/) and the ACRONYM grasps [here](https://sites.google.com/nvidia.com/graspdataset).

Next, clone Manifold library and install it as follows.
```
mkdir /path/to/lsmp/extern
git clone https://github.com/hjwdzh/Manifold.git /path/to/lsmp/extern/Manifold
./scripts/install_manifold.sh
```


Then, run the following script to preprocess Shapenet objects:
```
mkdir -p ~/projects/lsmp/datasets/shapenet
python -m core.data.gazebo.scripts.generate_acronym_dataset /path/to/shapenetsem/meshes /path/to/acronym datasets/shapenet
```

Finally, use the script below to convert OBJ file to SDF file in Shapenet datasets
```
python -m core.data.gazebo.scripts.obj2sdf
```

## Set up the conveyor plugin

(In `catkin_ws/src`)

```
git clone https://github.com/rokokoo/gazebo-conveyor.git
```

To set the power value, you need to use rosservice.

```
rosservice call /robot_ns/conveyor/control "power: 15.0"
```
(Power range: [0, 100])

## Run the code

### To generate init config

This generates an init config file that contains the pose of the objects randomly spawned in a limited range.

In a terminal,

```
export GAZEBO_MODEL_PATH=~/projects/lsmp/datasets/shapenet/
export GAZEBO_RESOURCE_PATH=~/projects/lsmp/core/data/gazebo/assets/worlds
roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch
```

In another terminal,

```
export DATA_DIR=~/projects/lsmp/datasets
python -m core.data.gazebo.scripts.generate_scene_configs
```

### To plan paths using an OMPL algorithm

This sets the scene from an init config file, outputs planned paths in a json file, and saves the success rate and planning time in a txt file.

In a terminal,

```
export GAZEBO_MODEL_PATH=~/projects/lsmp/datasets/shapenet/
export GAZEBO_RESOURCE_PATH=~/projects/lsmp/core/data/gazebo/assets/worlds
roslaunch panda_moveit_config panda_shapenet_empty_camera_connected.launch
```

In another terminal,

```
python -m core.data.gazebo.scripts.evaluate_motion_planner
```

### Old launch files without Panda in Gazebo

Remember to set `use_fake_controller=True` in `PandaSimClient`.

```
roslaunch panda_moveit_config shapenet_empty_camera_connected.launch
roslaunch panda_moveit_config shapenet_empty_camera_unconnected.launch
```

## Unity demo

To visualise the planned paths, upload the json file to this demo:
https://ascane.github.io/assets/portfolio/panda-demo-web-v2/

## Misc

- Install missing moveit_simple_controller_manager

First, check if moveit_simple_controller_manager is installed.

```
rospack find moveit_simple_controller_manager
```

If not, do a sparse checkout from moveit and build it.

```
catkin_ws/src$ mkdir moveit
cd moveit
git init
git remote add -f origin https://github.com/ros-planning/moveit.git
git config core.sparseCheckout true
echo "moveit_plugins/moveit_simple_controller_manager/" >> .git/info/sparse-checkout
git pull origin noetic-devel
cd../..
catkin_ws$ catkin_make
rosdep install -y --from-paths src --ignore-src --rosdistro noetic
```

- Install missing moveit_planners_chomp

Check if moveit_planners_chomp is installed.

```
rospack find moveit_planners_chomp
```

Build it from source.

```
echo "moveit_planners/" >> .git/info/sparse-checkout
git pull origin noetic-devel
cd../..
catkin_ws$ catkin_make
```

- The default planning time is set to 1 second in CHOMP, which can be changed in `panda_moveit_config/config/chomp_planning.yaml`.

- The default planner used is RRTConnect. To use another OMPL planner, edit `panda_moveit_config/config/ompl_planning.yaml` and add a `default_planner_config`. For example,

```
<...>

panda_arm:
  default_planner_config: RRTstarkConfigDefault
  planner_configs:
    - SBLkConfigDefault
    - ESTkConfigDefault
    - LBKPIECEkConfigDefault
    - BKPIECEkConfigDefault
    - KPIECEkConfigDefault
    - RRTkConfigDefault
    - RRTConnectkConfigDefault
    - RRTstarkConfigDefault
    - TRRTkConfigDefault
    <...>
```

- To debug Gazebo world, launch with verbose flag. For example,
```
roslaunch panda_moveit_config demo_gazebo_shapenet_scaled.launch verbose:=True
```

- [Use a Gazebo Depth Camera with ROS](https://classic.gazebosim.org/tutorials?tut=ros_depth_camera&cat=connect_ros).
This is already done in the camera model and the world file.

Make sure `gazebo_ros_pkgs` is installed as in the [prerequisites](https://classic.gazebosim.org/tutorials?tut=ros_installing&cat=connect_ros). Otherwise, you would see `libgazebo_ros_openni_kinect.so not found` in verbose mode.

Issue encountered when installing gazebo_ros_pkgs: failed to fetch the packages.

Solution:
[link1](https://answers.ros.org/question/326486/error-installing-ros-kinetic-gazebo-ros-control/), 
[link2](https://answers.ros.org/question/325039/apt-update-fails-cannot-install-pkgs-key-not-working/)

- Shapenet models use a different coordinate system (Y-axis=up). Thus the fixed orientation in `ModelStateSetter`.
```
self.set_state(obj_name, px, py, pz, w=0.5, x=0.5, y=0.5, z=0.5)
```

- The point cloud from the kinect camera appears to be rotated. A transform is added in the launch files to counter this effect.
```
<node pkg="tf2_ros" type="static_transform_publisher" name="to_camera_rot" args="0 0 0 -1.57 0 -1.57  camera_base camera_link" />
```

- [Moveit tutorial](http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/perception_pipeline/perception_pipeline_tutorial.html) for using depth image to generate Octomap in the planning scene.
