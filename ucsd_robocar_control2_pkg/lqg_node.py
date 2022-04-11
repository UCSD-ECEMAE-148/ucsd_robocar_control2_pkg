import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
from sensor_msgs.msg import IMU
import tf
from tf.transformations import euler_from_quaternion
from .controller_submodule.lqr_calculator import LQRDesign
from .controller_submodule.car_model import CarModel
from .state_estimate_submodule.linear_kalman_calculator import LinearKalmanFilter
import time
import math
import numpy as np
import os

NODE_NAME = 'lqg_node'
ACTUATOR_TOPIC_NAME = '/cmd_vel'

POSE_TOPIC_NAME = '/pose'
PATH_TOPIC_NAME = '/path'
ERROR_TOPIC_NAME = '/path_error'
IMU_TOPIC_NAME = '/razor/imu'


class LqgController(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.twist_publisher = self.create_publisher(Twist, ACTUATOR_TOPIC_NAME, 10)
        self.twist_cmd = Twist()

        # Get sensor measurements
        # Get IMU measurement
        self.velocity_subscriber = self.create_subscription(IMU, IMU_TOPIC_NAME, self.imu_measurement, 10)
        self.Ts = 100  # imu sample frequency (Hz)
        self.vx = 0
        self.vy = 0

        # Get GPS/Lidar measurements
        self.pose_subscriber = self.create_subscription(Pose, POSE_TOPIC_NAME, self.pose_measurement, 10)
        self.pose_subscriber
        self.path_subscriber = self.create_subscription(Path, PATH_TOPIC_NAME, self.set_path, 10)
        self.path_subscriber

        # Get road marker error measurements from camera
        self.pose_error_subscriber = self.create_subscription(Float32MultiArray, ERROR_TOPIC_NAME,
                                                              self.camera_measurment, 10)
        self.pose_error_subscriber

        # Controller modules
        self.car_model = CarModel()
        self.lqr_calc = LQRDesign()
        self.kalman_calc = LinearKalmanFilter()
        self.P = np.diag([1, 1, 1, 1])
        self.Qo = np.diag([1, 1, 1, 1])
        self.Ro = [0.1]
        self.x0 = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.state_measurement = self.x0
        self.state_est = self.x0
        self.u = 0

        # Sensor measurements
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.roll_rate = 0
        self.pitch_rate = 0
        self.yaw_rate = 0
        self.ax = 0
        self.ay = 0
        self.az = 0

        # Path coordinates
        self.x_path = []
        self.y_path = []
        self.z_path = []
        self.roll_path = []
        self.pitch_path = []
        self.yaw_path = []

        # Calculated states
        self.ecg = 0  # cross-track error
        self.theta_e = 0  # heading error
        self.theta_e_dot = 0  # heading error rate

        # Call controller
        self.create_timer(self.Ts, self.controller)

        # Declare ROS parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('error_threshold', 0.15),
                ('zero_throttle', 0.0),
                ('max_throttle', 0.2),
                ('min_throttle', 0.1),
                ('max_right_steering', 1.0),
                ('max_left_steering', -1.0)
            ])
        self.error_threshold = self.get_parameter('error_threshold').value  # between [0,1]
        self.zero_throttle = self.get_parameter('zero_throttle').value  # between [-1,1] but should be around 0
        self.max_throttle = self.get_parameter('max_throttle').value  # between [-1,1]
        self.min_throttle = self.get_parameter('min_throttle').value  # between [-1,1]
        self.max_right_steering = self.get_parameter('max_right_steering').value  # between [-1,1]
        self.max_left_steering = self.get_parameter('max_left_steering').value  # between [-1,1]

        self.get_logger().info(
            f'\nerror_threshold: {self.error_threshold}'
            f'\nzero_throttle: {self.zero_throttle}'
            f'\nmax_throttle: {self.max_throttle}'
            f'\nmin_throttle: {self.min_throttle}'
            f'\nmax_right_steering: {self.max_right_steering}'
            f'\nmax_left_steering: {self.max_left_steering}'
        )

    def imu_measurement(self, imu_data):

        # TODO: what is frequency of data coming in?

        quaternion = (imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)

        # FIXME: confirm coordinate axes
        # orientation
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]

        # angular velocity
        self.roll_rate = imu_data.angular_velocity.x
        self.pitch_rate = imu_data.angular_velocity.y
        self.yaw_rate = imu_data.angular_velocity.z

        # linear acceleration
        self.ax = imu_data.linear_acceleration.x
        self.ay = imu_data.linear_acceleration.y
        self.az = imu_data.linear_acceleration.z

        # linear velocity
        self.vx = self.vx + (self.ax * self.Ts)
        self.vy = self.vy + (self.ay * self.Ts)

    def pose_measurement(self, pose_data):

        # TODO: what is frequency of data coming in?
        # FIXME: confirm coordinate axes
        # car coordinates
        self.x = pose_data.position.x
        self.y = pose_data.position.y
        self.z = pose_data.position.z

    def set_path(self, path_data):
        quaternion = (path_data.orientation.x, path_data.orientation.y, path_data.orientation.z, path_data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)

        # FIXME: confirm coordinate axes
        # path orientation
        self.roll_path = euler[0]
        self.pitch_path = euler[1]
        self.yaw_path = euler[2]

        # path coordinates
        self.x_path = path_data.position.x
        self.y_path = path_data.position.y
        self.z_path = path_data.position.z

    def camera_measurment(self, error_data):
        # TODO: what is frequency of data coming in?
        self.ecg = error_data.data[0]  # cross-track error (pose_error_y * delta_x_path - pose_error_x * delta_y_path) / (delta_x_path^2 + delta_y_path^2)
        self.theta_e = error_data.data[1]  # theta_e = path_angle - car_yaw_angle

    def update_states(self):
        # NON-LINEAR SENSOR MODEL: NEED EKF
        delta_x_path = self.x_path[1] - self.x_path[0]
        delta_y_path = self.y_path[1] - self.y_path[0]
        pose_error_x = self.x - self.x_path[0]
        pose_error_y = self.y - self.y_path[0]
        car_heading = (self.yaw_imu + self.yaw_pose_measurement) / 2
        theta_e_km1 = self.state_measurement[0][2]
        theta_e_k = self.yaw_path[0] - car_heading
        self.state_measurement[0][0] = (pose_error_y * delta_x_path - pose_error_x * delta_y_path) / (delta_x_path ** 2 + delta_y_path ** 2)
        self.state_measurement[0][2] = theta_e_k
        self.state_measurement[0][1] = self.vy + self.vx * math.sin(theta_e_k)
        self.state_measurement[0][3] = (theta_e_k - theta_e_km1) / self.Ts

    def controller(self):
        """
        sensor measurements:
        -imu and pose data

        reference tracking:
        -path data

        states:
        x1 - ecg: cross-trackk error from center of gravity (cg) --- = (pose_error_y * delta_x_path - pose_error_x * delta_y_path) / (delta_x_path^2 + delta_y_path^2)
        x2 - ecg_dot: cross-trackk error rate from cg --- = vy + vx * sin(theta_error)
        x3 - theta_e: heading error --- = path_angle - car_yaw_angle
        x4 - theta_e_dot: heading error rate --- = (theta_e_k - theta_e_km1) / self.Ts # theta_e_k = heading error at sample k AND theta_e_km1 = heading error at sample k - 1
        """

        # Throttle scheduling (function of error)
        inf_throttle = self.min_throttle - (self.min_throttle - self.max_throttle) / (1 - self.error_threshold)
        throttle_float_raw = ((self.min_throttle - self.max_throttle) / (1 - self.error_threshold)) * abs(self.ek) + inf_throttle
        throttle_float = self.clamp(throttle_float_raw, self.max_throttle, self.min_throttle)

        # Update Car model LTV system --- A(Vx)
        sys = self.car_model.build_error_model(self.vx)

        # Get gains
        K = self.lqr_calc.compute_single_gain_sample(sys)

        # apply control to excite system u = -K * X_est
        steering_float_raw = -np.dot(K[0], self.state_est).flat[0]
        self.u = self.clamp(steering_float_raw, self.max_right_steering, self.max_left_steering)

        # Get Current Measurement
        self.y = self.car_model.calc_output(self.state_measurement)

        # Get optimal state estimates
        self.state_est, self.P = self.kalman_calc.lkf(sys, self.state_est, self.u, self.y, self.P, self.Qo, self.Ro)

        # Get new sensor measurements
        self.update_states()

        # Publish values
        try:
            # publish control signals
            self.twist_cmd.angular.z = self.u
            self.twist_cmd.linear.x = throttle_float
            self.twist_publisher.publish(self.twist_cmd)
        except KeyboardInterrupt:
            self.twist_cmd.linear.x = self.zero_throttle
            self.twist_publisher.publish(self.twist_cmd)

    def clamp(self, value, upper_bound, lower_bound=None):
        if lower_bound is None:
            lower_bound = -upper_bound  # making lower bound symmetric about zero
        if value < lower_bound:
            value_c = lower_bound
        elif value > upper_bound:
            value_c = upper_bound
        else:
            value_c = value
        return value_c


def main(args=None):
    rclpy.init(args=args)
    lqg_publisher = LqgController()
    try:
        rclpy.spin(lqg_publisher)
        lqg_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        lqg_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        lqg_publisher.twist_cmd.linear.x = lqg_publisher.zero_throttle
        lqg_publisher.twist_publisher.publish(lqg_publisher.twist_cmd)
        time.sleep(1)
        lqg_publisher.destroy_node()
        rclpy.shutdown()
        lqg_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()
