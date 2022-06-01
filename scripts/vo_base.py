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
from sfm_diff_drive.msg import (
    SFMDriveFeedback,
    SFMDriveResult,
    SFMDriveAction,
)
from move_base_msgs.msg import MoveBaseAction
from actionlib_msgs.msg import GoalID
from nav_msgs.msg import OccupancyGrid


class VODrive(object):
    def __init__(self):

        # initial variables
        self.agents_states_register = []
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.obstacles_pos = []

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

    def agents_state_callback(self, data):
        """
        callback para obtener lista de info de agentes
        """
        self.agents_states_register = data.agent_states

    def robot_pos_callback(self, data):
        """
        callback para agarrar datos de posicion del robot
        """
        data_position = data.pose.pose.position
        self.robot_position = np.array(
            [data_position.x, data_position.y, data_position.z], np.dtype("float64")
        )

        self.robot_orientation = np.array(
            [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w,
            ],
            np.dtype("float64"),
        )

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
