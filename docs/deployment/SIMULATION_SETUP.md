# Simulation setup

Use Ubuntu 24.04 (native or WSL 2) with ROS 2 Jazzy and Gazebo Harmonic
installed and the ROS apt repository configured. From the repository root,
install the tools and the Jackal description.

```bash
sudo apt install git python3-colcon-common-extensions python3-rosdep python3-numpy \
  ros-jazzy-ros-gz ros-jazzy-clearpath-platform-description
source /opt/ros/jazzy/setup.bash
mkdir -p go2_gait_ws/src
git clone https://github.com/abutalipovvv/go2_ros2_sim_py.git go2_gait_ws/src/go2_ros2_sim_py
git -C go2_gait_ws/src/go2_ros2_sim_py checkout 92d6bc99af555e332bb727c6effc482e5dc30fda
git -C go2_gait_ws/src/go2_ros2_sim_py apply "$PWD/patches/go2-forward-stride.patch"
```

Initialize rosdep once if it has not been initialized (`sudo rosdep init`).
Build the gait overlay. The standalone legacy description packages are
excluded from this build because they declare obsolete Gazebo dependencies.

```bash
cd go2_gait_ws
rosdep update
rosdep install --from-paths src/go2_ros2_sim_py/{gazebo_sim,quadropted_controller,quadropted_msgs} \
  --ignore-src --rosdistro jazzy -y
colcon build --symlink-install --packages-select gazebo_sim quadropted_controller quadropted_msgs
cd ..
bash scripts/run_mixed_realtime.sh nominal
```

See [quick start](../../GETTING_STARTED.md) for the other profiles, log files
and training commands. The optional original generic MAPPO experiments use the pinned on-policy
submodule. Initialize it from the repository root with
`git submodule update --init external/on-policy`.
