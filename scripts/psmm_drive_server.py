#!/usr/bin/env python3
import rospy

from pedsim_msgs.msg import AgentStates, AgentGroups
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf import TransformListener
import tf
import numpy as np
import math
import actionlib

from psmm_drive.msg import (
    PSMMDriveFeedback,
    PSMMDriveResult,
    PSMMDriveAction,
)

from geometry_msgs.msg import Point

from actionlib_msgs.msg import GoalID


class ProactiveSocialMotionModelDriveAction(object):

    _feedback = PSMMDriveFeedback()
    _result = PSMMDriveResult()

    def __init__(self):

        # base variables
        self.goal_set = False

        self.xy_tolerance = 1

        self._action_name = "psmm_drive_node"

        self.agents_states_register = []
        self.agents_groups_register = []
        self.current_waypoint = np.array([0, 0, 0], np.dtype("float64"))
        self.robot_position = np.array([0, 0, 0], np.dtype("float64"))

        self.robot_orientation = np.array([0, 0, 0, 0], np.dtype("float64"))

        self.robot_current_vel = np.array([0, 0, 0], np.dtype("float64"))
        self.relaxation_time = 0.5
        self.laser_ranges = np.zeros(360)

        self.hrvo_vel = np.array([0, 0, 0], np.dtype("float64"))

        self.walls_range = []

        # nearest obstacle
        self.nearest_obstacle = np.array(
            [
                0,
                0,
                0,
            ],
            np.dtype("float64"),
        )

        self.agent_radius = 1
        self.force_sigma_obstacle = 0.8

        # for social force computing

        self.lambda_importance = 2
        self.gamma = 0.35
        self.n = 2
        self.n_prime = 3

        # constants for forces and other parameters
        self.force_factor_desired = rospy.get_param("~force_desired", 4.2)
        self.force_factor_social = rospy.get_param("~force_social", 3.64)
        self.force_factor_obstacle = rospy.get_param("~force_obstacle", 35)
        self.robot_max_vel = rospy.get_param("~max_vel", 0.4)
        self.robot_max_turn_vel = rospy.get_param("~max_vel_turn", 0.4)
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        self.waypoints = rospy.get_param("~waypoints", [])
        self.using_waypoints = False
        self.map = OccupancyGrid()

        # topic configs
        self.global_plan_topic = rospy.get_param("~goal_path_topic", "")
        self.agent_states_topic = rospy.get_param(
            "~social_agents_topic", "/pedsim_simulator/simulated_agents"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/pepper/odom_groundtruth")
        self.laser_topic = rospy.get_param("~laser_topic", "/scan_filtered")
        self.map_topic = rospy.get_param("~map_topic", "/projected_map")

        self.current_goal_topic = rospy.get_param(
            "~current_goal_topic", "/psmm_current_goal"
        )

        self.tf = TransformListener()

        self._as = actionlib.SimpleActionServer(
            self._action_name,
            PSMMDriveAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )
        self._as.start()

        #! subscribers
        self.agents_states_subs = rospy.Subscriber(
            self.agent_states_topic,
            AgentStates,
            self.agents_state_callback,
        )

        self.agents_groups_subs = rospy.Subscriber(
            "/pedsim_simulator/simulated_groups",
            AgentGroups,
            self.agents_groups_callback,
        )

        self.robot_pos_subs = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.robot_pos_callback,
        )

        self.laser_scan_subs = rospy.Subscriber(
            self.laser_topic, LaserScan, self.laser_scan_callback
        )

        self.obstacles_subs = rospy.Subscriber(
            self.map_topic, OccupancyGrid, self.map_callback
        )

        if self.global_plan_topic != "":
            self.global_plan_sub = rospy.Subscriber(
                self.global_plan_topic, Path, self.global_plan_callback, queue_size=1
            )

        self.hrvo_subs = rospy.Subscriber("/cmd_vel_hrvo", Twist, self.hrvo_vel_cb)

        #! publishers
        self.velocity_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)

        self.current_goal_pub = rospy.Publisher(
            self.current_goal_topic, Point, queue_size=10
        )

        self.goal_achieved_pub = rospy.Publisher("/goal_achieved", Bool, queue_size=10)

    def global_plan_callback(self, msg):
        self.waypoints = []
        for pose in msg.poses:
            self.waypoints.append([pose.pose.position.x, pose.pose.position.y])
        self.obstacle_map_processing()

    def check_goal_reached(self):
        if (
            abs(
                np.linalg.norm(
                    np.array(
                        [self.current_waypoint[0], self.current_waypoint[1]],
                        np.dtype("float64"),
                    )
                    - np.array(
                        [self.robot_position[0], self.robot_position[1]],
                        np.dtype("float64"),
                    )
                )
            )
            <= 0.2
        ):
            if self.using_waypoints:
                if len(self.waypoints) != 1:
                    self.waypoints.pop(0)
                    self.current_waypoint = np.array(
                        [self.waypoints[0][0], self.waypoints[0][1], 0],
                        np.dtype("float64"),
                    )
                    current_goal_msg = Point()
                    current_goal_msg.x = self.current_waypoint[0]
                    current_goal_msg.y = self.current_waypoint[1]
                    current_goal_msg.z = 0

                    self.current_goal_pub.publish(current_goal_msg)
                    print("NEXT WAYPOINT NOW: ", len(self.waypoints))

                else:
                    return True
            else:
                return True
        return False

    # * callbacks

    # hrvo velocity

    def hrvo_vel_cb(self, data: Twist):
        self.hrvo_vel[0] = data.linear.x
        self.hrvo_vel[1] = data.linear.y

    def execute_cb(self, goal):
        rospy.loginfo("Starting social drive")
        r_sleep = rospy.Rate(30)
        cancel_move_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        cancel_msg = GoalID()
        cancel_move_pub.publish(cancel_msg)

        self.goal_set = True

        if len(self.waypoints) > 0:
            self.current_waypoint = np.array(
                [self.waypoints[0][0], self.waypoints[0][1], 0], np.dtype("float64")
            )
            self.using_waypoints = True
        else:
            self.current_waypoint = np.array(
                [goal.goal.x, goal.goal.y, goal.goal.z], np.dtype("float64")
            )

        current_goal_msg = Point()
        current_goal_msg.x = self.current_waypoint[0]
        current_goal_msg.y = self.current_waypoint[1]
        current_goal_msg.z = 0

        self.current_goal_pub.publish(current_goal_msg)

        while not self.check_goal_reached():

            obstacle_complete_force = (
                self.force_factor_obstacle * self.obstacle_force_walls()
            )

            social_complete_force = self.force_factor_social * self.social_force()

            desired_complete_force = self.force_factor_desired * self.desired_force()

            complete_force = (
                desired_complete_force + social_complete_force + obstacle_complete_force
            )
            # print("#######################")

            # print("hrvo vel:", self.hrvo_vel)
            # print("desired force:", desired_complete_force)
            # print("nearest_obstacle:", self.nearest_obstacle)
            # print("obstacle force:", obstacle_complete_force)
            # print("complete force:", complete_force)

            # time.sleep(1)

            self.robot_current_vel = self.robot_current_vel + (complete_force / 25)

            speed = np.linalg.norm(self.robot_current_vel)

            if speed > self.robot_max_vel:
                self.robot_current_vel = (
                    self.robot_current_vel
                    / np.linalg.norm(self.robot_current_vel)
                    * self.robot_max_vel
                )

            quaternion = (
                self.robot_orientation[0],
                self.robot_orientation[1],
                self.robot_orientation[2],
                self.robot_orientation[3],
            )

            euler = tf.transformations.euler_from_quaternion(quaternion)

            robot_offset_angle = euler[2]

            if robot_offset_angle < 0:
                robot_offset_angle = 2 * math.pi + robot_offset_angle

            angulo_velocidad = math.atan2(
                self.robot_current_vel[0], self.robot_current_vel[1]
            )

            if angulo_velocidad > 0 and angulo_velocidad < (math.pi / 2):
                angulo_velocidad = (math.pi / 2) - angulo_velocidad
            elif angulo_velocidad > (math.pi / 2):
                angulo_velocidad = (2 * math.pi) - angulo_velocidad + (math.pi / 2)
            elif angulo_velocidad < 0:
                angulo_velocidad = (math.pi / 2) - angulo_velocidad
            elif angulo_velocidad == 0:
                angulo_velocidad = math.pi / 2
            elif abs(angulo_velocidad) == (math.pi / 2):
                angulo_velocidad = math.pi * 3 / 2

            if robot_offset_angle > (angulo_velocidad + math.pi):
                yaw_error = angulo_velocidad + 2 * math.pi - robot_offset_angle
            elif angulo_velocidad > (robot_offset_angle + math.pi):
                yaw_error = robot_offset_angle + 2 * math.pi - angulo_velocidad
            else:
                yaw_error = robot_offset_angle - angulo_velocidad

            yaw_error = -robot_offset_angle + angulo_velocidad

            if yaw_error < -math.pi:
                yaw_error = 2 * math.pi + yaw_error
            elif yaw_error > math.pi:
                yaw_error = -2 * math.pi + yaw_error

            if abs(yaw_error) < 0.2:
                w = 0
            else:
                w = yaw_error * self.robot_max_turn_vel

            if abs(w) > self.robot_max_turn_vel:
                if w > 0:
                    w = self.robot_max_turn_vel
                elif w < 0:
                    w = -self.robot_max_turn_vel

            if abs(yaw_error) > 1.3:
                vx = 0
            else:
                vx = np.linalg.norm(self.robot_current_vel) * math.cos(yaw_error)

            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = vx
            cmd_vel_msg.angular.z = w

            self.velocity_pub.publish(cmd_vel_msg)

            self._feedback.feedback = "robot moving"
            # rospy.loginfo("robot_moving")
            self._as.publish_feedback(self._feedback)
            r_sleep.sleep()
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0
        cmd_vel_msg.angular.z = 0

        self.velocity_pub.publish(cmd_vel_msg)
        self._result.result = "waypoint reached"
        rospy.loginfo("waypoint reached")
        self._as.set_succeeded(self._result)

        self.goal_achieved_pub.publish(True)

    # define MAP_INDEX(map, i, j) ((i) + (j) * map.size_x)
    def map_index(self, size_x, i, j):
        return i + j * size_x

    # define MAP_WXGX(map, i) (map.origin_x + (i - map.size_x / 2) * map.scale)

    def map_wx(self, origin_x, size_x, scale, i):
        return origin_x + (i - size_x / 2) * scale

    def map_wy(self, origin_y, size_y, scale, j):
        return origin_y + (j - size_y / 2) * scale

    def map_callback(self, data):
        self.map = data

    def obstacle_map_processing(self):

        cur_nearest_obs = (0, 0)
        cur_nearest_dist = 1000000000

        map_size_x = self.map.info.width
        map_size_y = self.map.info.height
        map_scale = self.map.info.resolution
        map_origin_x = self.map.info.origin.position.x + (map_size_x / 2) * map_scale
        map_origin_y = self.map.info.origin.position.y + (map_size_y / 2) * map_scale

        for j in range(0, map_size_y, 10):
            for i in range(0, map_size_x, 10):
                if self.map.data[self.map_index(map_size_x, i, j)] == 100:
                    w_x = self.map_wx(map_origin_x, map_size_x, map_scale, i)
                    w_y = self.map_wy(map_origin_y, map_size_y, map_scale, j)
                    cur_dist = np.power(w_x - self.robot_position[0], 2) + np.power(
                        w_y - self.robot_position[1], 2
                    )

                    if cur_dist < cur_nearest_dist:
                        cur_nearest_dist = cur_dist
                        cur_nearest_obs = (w_x, w_y)
                        # print(cur_dist)

        self.nearest_obstacle[0] = cur_nearest_obs[0]
        self.nearest_obstacle[1] = cur_nearest_obs[1]

    def map_index(self, size_x, i, j):
        return i + j * size_x

    # define MAP_WXGX(map, i) (map.origin_x + (i - map.size_x / 2) * map.scale)

    def map_wx(self, origin_x, size_x, scale, i):
        return origin_x + (i - size_x / 2) * scale

    def map_wy(self, origin_y, size_y, scale, j):
        return origin_y + (j - size_y / 2) * scale

    def laser_scan_callback(self, data):
        """
        callback para agarrar los datos del laser
        """
        self.laser_ranges = data.ranges

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

    def agents_state_callback(self, data):
        """
        callback para obtener lista de info de agentes
        """
        self.agents_states_register = data.agent_states

    def agents_groups_callback(self, data):
        """
        callback para obtener los datos de grupos de agentes
        """
        self.agents_groups_register = data

        # * force functions
        """
        funcion para obtener la fuerza al waypoint
        """

    def desired_force(self):

        desired_force = (self.hrvo_vel - self.robot_current_vel) / self.relaxation_time
        return desired_force

    def obstacle_force_walls(self):
        """
        funcion para obtener la fuerza de el obstaculo mas cercano conociendo la posicion exacta de todos ellos de manera estatica
        """

        diff_robot_obstacle = np.sqrt(
            np.power(self.nearest_obstacle[0] - self.robot_position[0], 2)
            + np.power(self.nearest_obstacle[1] - self.robot_position[1], 2)
        )

        nearest_obstacle_temp = self.robot_position - self.nearest_obstacle

        obstacle_vec_norm = np.linalg.norm(nearest_obstacle_temp)
        if obstacle_vec_norm != 0:
            norm_obstacle_direction = nearest_obstacle_temp / obstacle_vec_norm
        else:
            norm_obstacle_direction = np.array([0, 0, 0], np.dtype("float64"))

        distance = diff_robot_obstacle - self.agent_radius
        force_amount = math.exp(-distance / self.force_sigma_obstacle)
        final_rep_force = force_amount * norm_obstacle_direction
        return final_rep_force
        # else:
        #     return np.array([0, 0, 0], np.dtype("float64"))

    def obstacle_force(self):
        """
        funcion para obtener la fuerza de el obstaculo mas cercano
        """
        diff_robot_laser = []
        # obtener valores de el laser sus distancias
        for i in range(0, 360):
            distance = math.sqrt(
                math.pow(self.laser_ranges[i] * math.cos(math.radians(i - 90)), 2)
                + math.pow(self.laser_ranges[i] * math.sin(math.radians(i - 90)), 2)
            )
            diff_robot_laser.append(distance)

        diff_robot_laser = np.array(diff_robot_laser, np.dtype("float64"))

        for i in range(0, 360):
            if diff_robot_laser[i] == np.nan:
                diff_robot_laser[i] = np.inf

        min_index = 0
        tmp_val = 1000
        for i in range(0, 360):
            if diff_robot_laser[i] < tmp_val and diff_robot_laser[i] != 0:
                tmp_val = diff_robot_laser[i]
                min_index = i

        if diff_robot_laser[min_index] < 1:
            laser_pos = -1 * np.array(
                [
                    self.laser_ranges[min_index]
                    * math.cos(math.radians(min_index - 180)),
                    self.laser_ranges[min_index]
                    * math.sin(math.radians(min_index - 180)),
                    0,
                ],
                np.dtype("float64"),
            )

            laser_vec_norm = np.linalg.norm(laser_pos)
            if laser_vec_norm != 0:
                norm_laser_direction = laser_pos / laser_vec_norm
            else:
                norm_laser_direction = np.array([0, 0, 0], np.dtype("float64"))

            distance = diff_robot_laser[min_index] - self.agent_radius
            force_amount = math.exp(-distance / self.force_sigma_obstacle)
            final_rep_force = force_amount * norm_laser_direction
            return final_rep_force
        else:
            return np.array([0, 0, 0], np.dtype("float64"))

    def social_force(self):
        """
        funcion para obtener la fuerzas sociales de los alrededores
        """

        force = np.array([0, 0, 0], np.dtype("float64"))

        for i in self.agents_states_register:
            diff_position = (
                np.array(
                    [
                        i.pose.position.x,
                        i.pose.position.y,
                        i.pose.position.z,
                    ],
                    np.dtype("float64"),
                )
                - self.robot_position
            )

            diff_direction = diff_position / np.linalg.norm(diff_position)

            agent_velocity = i.twist.linear
            diff_vel = self.robot_current_vel - np.array(
                [
                    agent_velocity.x,
                    agent_velocity.y,
                    agent_velocity.z,
                ],
                np.dtype("float64"),
            )

            interaction_vector = self.lambda_importance * diff_vel + diff_direction

            interaction_length = np.linalg.norm(interaction_vector)

            interaction_direction = interaction_vector / interaction_length

            # theta = angle(interaction_direction, diff_direction)

            theta = math.atan2(diff_direction[1], diff_direction[0]) - math.atan2(
                interaction_direction[1], interaction_direction[0]
            )

            B = self.gamma * interaction_length

            force_velocity_amount = -math.exp(
                -np.linalg.norm(diff_position) / B
                - (self.n_prime * B * theta) * (self.n_prime * B * theta)
            )

            force_angle_amount = -number_sign(theta) * math.exp(
                -np.linalg.norm(diff_position) / B
                - (self.n * B * theta) * (self.n * B * theta)
            )

            force_velocity = force_velocity_amount * interaction_direction

            force_angle = force_angle_amount * np.array(
                [
                    -interaction_direction[1],
                    interaction_direction[0],
                    0,
                ],
                np.dtype("float64"),
            )

            force += force_velocity + force_angle
        return force


"""
obtencion de signo a partir de una numero
"""


def number_sign(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    return -1


"""
producto punto de dos vectores
"""


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


"""
modulo de un vector
"""


def length(v):
    return math.sqrt(dotproduct(v, v))


"""
angulo entre dos vectores
"""


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


if __name__ == "__main__":
    rospy.init_node("psmm_drive_node")
    server = ProactiveSocialMotionModelDriveAction()
    rospy.spin()
