cmake_minimum_required(VERSION 3.0.2)
project(psmm)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sfm_diff_drive
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES psmm
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

