#!/bin/bash

# Kill any leftover processes
killall gzserver gzclient 2>/dev/null
pkill -f ros2 2>/dev/null
sleep 3

# Source ROS2
source /opt/ros/humble/setup.bash
source ~/go2_ws/install/setup.bash

# Launch Go2 in hospital world
ros2 launch go2_config gazebo.launch.py \
  world:=/home/msi/.gazebo/worlds/hospital.world \
  world_init_x:=0.0 \
  world_init_y:=0.0 \
  world_init_z:=0.6