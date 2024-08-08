import rclpy
from rclpy.node import Node

from pedsim_msgs.msg import AgentStates
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, Point
import numpy as np
import math
from math import sqrt, cos, sin, atan2, asin, pi as PI
from tf_transformations import euler_from_quaternion


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


def wrapAngle(angle):
    """wrapAngle
    Calculates angles values between 0 and 2pi"""
    return angle + (2.0 * math.pi * math.floor((math.pi - angle) / (2.0 * math.pi)))


def intersect(pA, vA, RVO_BA_all):
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
    if suitable_V:
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


class VODrive(Node):
    def __init__(self):
        super().__init__("vo_node")

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
        self.max_vel = 0.4
        self.robot_radius = 0.35
        self.agent_radius = 0.40
        self.obstacle_radius = 0.1

        self.goal_available = False

        # topic configs
        self.global_plan_topic = self.declare_parameter("goal_path_topic", "").value
        self.agent_states_topic = self.declare_parameter(
            "social_agents_topic", "/pedsim_simulator/simulated_agents"
        ).value
        self.odom_topic = self.declare_parameter(
            "odom_topic", "/pepper/odom_groundtruth"
        ).value
        self.laser_topic = self.declare_parameter("laser_topic", "/scan_filtered").value
        self.map_topic = self.declare_parameter("map_topic", "/projected_map").value

        #! subcribers
        self.goal_location_sub = self.create_subscription(
            Point, "/psmm_current_goal", self.goal_callback, 10
        )

        self.agents_states_subs = self.create_subscription(
            AgentStates, self.agent_states_topic, self.agents_state_callback, 10
        )

        self.robot_pos_subs = self.create_subscription(
            Odometry, self.odom_topic, self.robot_pos_callback, 10
        )

        self.obstacles_subs = self.create_subscription(
            OccupancyGrid, self.map_topic, self.obstacle_map_processing, 10
        )

        #! publishers
        self.vo_cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel_hrvo", 10)

        # Timer for running the node
        # self.timer = self.create_timer(0.01, self.run)

    #! CALLBACKS

    def goal_callback(self, goal: Point):
        self.goal[0] = goal.x
        self.goal[1] = goal.y
        self.goal_available = True

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

    def obstacle_map_processing(self, data: OccupancyGrid):
        self.obstacles_pos = []

        map_size_x = data.info.width
        map_size_y = data.info.height
        map_scale = data.info.resolution
        map_origin_x = data.info.origin.position.x + (map_size_x / 2) * map_scale
        map_origin_y = data.info.origin.position.y + (map_size_y / 2) * map_scale

        for j in range(0, map_size_y, 10):
            for i in range(0, map_size_x, 10):
                # print("map size: ", map_size_x * map_size_y)
                if data.data[self.map_index(map_size_x, i, j)] == 100:
                    w_x = self.map_wx(map_origin_x, map_size_x, map_scale, i)
                    w_y = self.map_wy(map_origin_y, map_size_y, map_scale, j)
                    self.obstacles_pos.append([w_x, w_y])

    def map_index(self, size_x, i, j):
        return i + j * size_x

    def map_wx(self, origin_x, size_x, scale, i):
        return origin_x + (i - size_x / 2) * scale

    def map_wy(self, origin_y, size_y, scale, j):
        return origin_y + (j - size_y / 2) * scale

    #! computing functions
    def compute_des_vel(self):
        v_des = [0, 0, 0]
        goal_vec = self.goal - self.robot_position
        # print("goal_vec:", goal_vec)
        norm = np.linalg.norm(goal_vec)
        # print("distance: ", norm)
        # print("norm:", norm)
        if norm != 0:
            v_des = self.max_vel * (goal_vec / norm)

        # if norm < 0.05:
        #     v_des = [0, 0, 0]
        # else:

        return v_des

    def vo_velocity(self):
        vA = [self.robot_velocities[0], self.robot_velocities[1]]
        pA = [self.robot_position[0], self.robot_position[1]]
        RVO_BA_all = []

        for agent in self.agents_states_register:
            vB = [agent.twist.linear.x, agent.twist.linear.y]
            pB = [agent.pose.position.x, agent.pose.position.y]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            if 2 * self.robot_radius > dist_BA:
                dist_BA = 2 * self.robot_radius
            theta_BAort = asin(2 * self.robot_radius / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            dist_dif = distance([0.5 * (vB[0] - vA[0]), 0.5 * (vB[1] - vA[1])], [0, 0])
            transl_vB_vA = [
                pA[0] + vB[0] + cos(theta_ort_left) * dist_dif,
                pA[1] + vB[1] + sin(theta_ort_left) * dist_dif,
            ]
            RVO_BA = [
                transl_vB_vA,
                bound_left,
                bound_right,
                dist_BA,
                2 * self.robot_radius,
            ]
            RVO_BA_all.append(RVO_BA)

        for i in range(0, len(self.obstacles_pos), 10):
            vB = [0, 0]
            pB = self.obstacles_pos[i]
            transl_vB_vA = [pA[0] + vB[0], pA[1] + vB[1]]
            dist_BA = distance(pA, pB)
            theta_BA = atan2(pB[1] - pA[1], pB[0] - pA[0])
            # over-approximation of square to circular
            OVER_APPROX_C2S = 1.5
            rad = self.obstacle_radius * OVER_APPROX_C2S
            if (rad + self.robot_radius) > dist_BA:
                dist_BA = rad + self.robot_radius
            theta_BAort = asin((rad + self.robot_radius) / dist_BA)
            theta_ort_left = theta_BA + theta_BAort
            bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
            theta_ort_right = theta_BA - theta_BAort
            bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
            RVO_BA = [
                transl_vB_vA,
                bound_left,
                bound_right,
                dist_BA,
                rad + self.agent_radius,
            ]
            RVO_BA_all.append(RVO_BA)
        vA_post = intersect(pA, self.compute_des_vel(), RVO_BA_all)
        # print(vA_post)
        return vA_post[:]

    def run(self):
        while rclpy.ok():
            self.vo_velocity_val = self.vo_velocity()
            # print("===================")
            # print("hrvo: ", self.vo_velocity_val)
            # print("goal:", self.goal)

            # orientation_list = [
            #     self.robot_orientation[0],
            #     self.robot_orientation[1],
            #     self.robot_orientation[2],
            #     self.robot_orientation[3],
            # ]
            # (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

            # angle = wrapAngle(
            #     math.atan2(self.vo_velocity_val[1], self.vo_velocity_val[0]) - yaw
            # )
            # print("angle: ", angle)
            # print("velocity: ", self.vo_velocity_val)
            self.vo_twist_msg.linear.x = self.vo_velocity_val[0]
            self.vo_twist_msg.linear.y = self.vo_velocity_val[1]
            # self.vo_twist_msg.angular.z = 0.5*(angle/3.14)
            # self.vo_twist_msg.angular.z = 0.1
            if self.goal_available:
                self.vo_cmd_vel_pub.publish(self.vo_twist_msg)
            rclpy.spin_once(self)


def main(args=None):
    rclpy.init(args=args)
    vo_node = VODrive()
    vo_node.run()
    vo_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
