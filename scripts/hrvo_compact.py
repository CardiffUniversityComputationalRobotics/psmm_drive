#!/usr/bin/env python3
import rospy

from pedsim_msgs.msg import AgentStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import math
import actionlib
from psmm_drive.msg import PSMMDriveActionGoal

from nav_msgs.msg import OccupancyGrid

from math import sqrt
from math import cos, sin, atan2, asin

from math import pi as PI
import time


def distance(pose1, pose2):
    """compute Euclidean distance for 2D"""
    return sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2) + 0.001


def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= PI:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left < 0) and (theta_right > 0):
            theta_left += 2 * PI
            if theta_dif < 0:
                theta_dif += 2 * PI
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left > 0) and (theta_right < 0):
            theta_right += 2 * PI
            if theta_dif < 0:
                theta_dif += 2 * PI
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False


def intersect(pA, vA, RVO_BA_all):
    # print '----------------------------------------'
    # print 'Start intersection test'
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    unsuitable_V = []
    for theta in np.arange(0, 2 * PI, 0.1):
        for rad in np.arange(0.02, norm_v + 0.02, norm_v / 5.0):
            new_v = [rad * cos(theta), rad * sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                suitable_V.append(new_v)
            else:
                unsuitable_V.append(new_v)
    new_v = vA[:]
    suit = True
    for RVO_BA in RVO_BA_all:
        p_0 = RVO_BA[0]
        left = RVO_BA[1]
        right = RVO_BA[2]
        dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
        theta_dif = atan2(dif[1], dif[0])
        theta_right = atan2(right[1], right[0])
        theta_left = atan2(left[1], left[0])
        if in_between(theta_right, theta_dif, theta_left):
            suit = False
            break
    if suit:
        suitable_V.append(new_v)
    else:
        unsuitable_V.append(new_v)
    # ----------------------
    if suitable_V:
        # print 'Suitable found'
        vA_post = min(suitable_V, key=lambda v: distance(v, vA))
        new_v = vA_post[:]
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0] + pA[0] - p_0[0], new_v[1] + pA[1] - p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
    else:
        # print 'Suitable not found'
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dist = RVO_BA[3]
                rad = RVO_BA[4]
                dif = [unsuit_v[0] + pA[0] - p_0[0], unsuit_v[1] + pA[1] - p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    small_theta = abs(theta_dif - 0.5 * (theta_left + theta_right))
                    if abs(dist * sin(small_theta)) >= rad:
                        rad = abs(dist * sin(small_theta))
                    big_theta = asin(abs(dist * sin(small_theta)) / rad)
                    dist_tg = abs(dist * cos(small_theta)) - abs(rad * cos(big_theta))
                    if dist_tg < 0:
                        dist_tg = 0
                    tc_v = dist_tg / distance(dif, [0, 0])
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc) + 0.001
        WT = 0.2
        vA_post = min(
            unsuitable_V, key=lambda v: ((WT / tc_V[tuple(v)]) + distance(v, vA))
        )
    return vA_post


class VODrive(object):
    def __init__(self):

        # initial variables
        self.agents_states_register = []
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_velocities = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_orientation = np.array([0, 0, 0, 0], np.dtype("float64"))
        self.obstacles_pos = []
        self.vo_velocity_val = [0, 0]

        # v0_vel msg
        self.vo_twist_msg = Twist()

        # goal location
        self.goal = [0, 0, 0]

        #! modifiable parameters
        self.max_vel = 0.3
        self.robot_radius = 0.35
        self.agent_radius = 0.40
        self.obstacle_radius = 0.025

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
        self.vo_cmd_vel_pub = rospy.Publisher("cmd_vel_vo", Twist, queue_size=10)

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

    #! computing functions
    def compute_des_vel(self):
        v_des = [0, 0, 0]
        goal_vec = self.goal - self.robot_position
        norm = np.sqrt(np.power(goal_vec[0], 2) + np.power(goal_vec[1], 2))
        if norm < 0.25:
            v_des = [0, 0, 0]
        else:
            vel_vec = self.max_vel * (goal_vec / norm)
            v_des = [vel_vec[0], vel_vec[1], vel_vec[2]]
        return v_des

    def vo_velocity(self):
        vA = [self.robot_velocities[0], self.robot_velocities[1]]
        pA = [self.robot_position[0], self.robot_position[1]]
        RVO_BA_all = []

        for obstacle_pos in self.obstacles_pos:
            vB = [0, 0]
            pB = obstacle_pos
            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            if 2 * self.robot_radius > dist_BA:
                dist_BA = 2 * self.robot_radius
            theta_BAort = asin(2 * self.robot_radius / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [
                transl_vB_vA,
                bound_left,
                bound_right,
                dist_BA,
                2 * self.robot_radius,
            ]
            RVO_BA_all.append(RVO_BA)

        # for obstacle_pos in self.obstacles_pos:
        #     vB = [0, 0]
        #     pB = obstacle_pos
        #     transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
        #     dist_BA = distance(pA, pB)
        #     theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
        #     # over-approximation of square to circular
        #     OVER_APPROX_C2S = 1.5
        #     rad = self.obstacle_radius * OVER_APPROX_C2S
        #     if (rad + self.robot_radius) > dist_BA:
        #         dist_BA = rad + self.robot_radius
        #     theta_BAort = asin((rad + self.robot_radius) / dist_BA)
        #     theta_ort_left = theta_BA + theta_BAort
        #     bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
        #     theta_ort_right = theta_BA - theta_BAort
        #     bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
        #     RVO_BA = [
        #         transl_vB_vA,
        #         bound_left,
        #         bound_right,
        #         dist_BA,
        #         rad + self.agent_radius,
        #     ]
        #     RVO_BA_all.append(RVO_BA)
        vA_post = intersect(pA, self.compute_des_vel(), RVO_BA_all)
        return vA_post[:]

    def run(self):
        while not rospy.is_shutdown():
            self.vo_velocity_val = self.vo_velocity()
            self.vo_twist_msg.linear.x = self.vo_velocity_val[0]
            self.vo_twist_msg.linear.y = self.vo_velocity_val[1]

            self.vo_cmd_vel_pub.publish(self.vo_twist_msg)
            time.sleep(0.01)


if __name__ == "__main__":
    rospy.init_node("vo_node")
    vo_node = VODrive()
    vo_node.run()
