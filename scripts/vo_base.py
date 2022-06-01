#!/usr/bin/env python3
import rospy

from pedsim_msgs.msg import AgentStates, AgentGroups
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf import TransformListener
import tf
import numpy as np
import math
import actionlib
from psmm_drive.msg import PSMMDriveActionGoal

from move_base_msgs.msg import MoveBaseAction
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import OccupancyGrid


class VODrive(object):
    def __init__(self):

        # initial variables
        self.agents_states_register = []
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_velocities = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_orientation = np.array([0, 0, 0, 0], np.dtype("float64"))
        self.obstacles_pos = []

        # goal location
        self.goal = [0, 0, 0]

        #! subcribers

        self.goal_location_sub = rospy.Subscriber(
            "/psmm_drive_node/goal",
            PSMMDriveActionGoal,
        )

        self.agents_states_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_agents_overwritten",
            AgentStates,
            self.agents_state_callback,
        )

        self.robot_pos_subs = rospy.Subscriber(
            "/pepper/odom_groundtruth",
            Odometry,
            self.robot_pos_callback,
        )

        self.obstacles_subs = rospy.Subscriber(
            "/projected_map", OccupancyGrid, self.obstacle_map_processing
        )

        #! publishers
        self.cmd_pub = rospy.Publisher("cmd_vel_vo", Twist, queue_size=10)

    #! CALLBACKS

    def goal_callback(self, goal: PSMMDriveActionGoal):
        self.goal[0] = goal.goal.goal.x
        self.goal[1] = goal.goal.goal.y
        self.goal[2] = goal.goal.goal.z

    def agents_state_callback(self, data: AgentStates):
        """
        callback para obtener lista de info de agentes
        """
        self.agents_states_register = data.agent_states

    def robot_pos_callback(self, data: Odometry):
        """
        callback para agarrar datos de posicion del robot
        """

        self.robot_position[0] = data.pose.pose.position.x
        self.robot_position[1] = data.pose.pose.position.y
        self.robot_position[2] = data.pose.pose.position.z

        self.robot_orientation[0] = data.pose.pose.orientation.x
        self.robot_orientation[1] = data.pose.pose.orientation.y
        self.robot_orientation[2] = data.pose.pose.orientation.z
        self.robot_orientation[3] = data.pose.pose.orientation.w

        self.robot_velocities[0] = data.twist.twist.linear.x
        self.robot_velocities[1] = data.twist.twist.linear.y
        self.robot_velocities[2] = 0

    def obstacle_map_processing(self, data):

        self.obstacles_pos = []

        map_size_x = data.info.width
        map_size_y = data.info.height
        map_scale = data.info.resolution
        map_origin_x = data.info.origin.position.x + (map_size_x / 2) * map_scale
        map_origin_y = data.info.origin.position.y + (map_size_y / 2) * map_scale

        for j in range(0, map_size_y):
            for i in range(0, map_size_x):
                if data.data[self.map_index(map_size_x, i, j)] == 100:
                    w_x = self.map_wx(map_origin_x, map_size_x, map_scale, i)
                    w_y = self.map_wy(map_origin_y, map_size_y, map_scale, j)
                    self.obstacles_pos.append([w_x, w_y])


if __name__ == "__main__":
    rospy.init_node("vo_node")
    server = VODrive()
    rospy.spin()
