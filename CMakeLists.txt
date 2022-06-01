cmake_minimum_required(VERSION 3.0.2)
project(psmm)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
actionlib
actionlib_msgs
message_generation
# roscpp
rospy
std_msgs
geometry_msgs
)

add_action_files(
  FILES
  PSMMDrive.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
  geometry_msgs
)

catkin_package(
  # INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS actionlib actionlib_msgs rospy std_msgs message_runtime geometry_msgs
 
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

