import os
import time
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from .controller_submodule.lqr_calculator import LQRDesign
from .controller_submodule.car_model import CarModel
from .controller_submodule.linear_kalman_filter import LinearKalmanFilter
from .controller_submodule.ss_simulation import StateSpaceSimulation
import numpy as np
import pandas as pd
import math
import time

NODE_NAME = 'lqg_w_node'
ACTUATOR_TOPIC_NAME = '/teleop'
# ACTUATOR_TOPIC_NAME = '/lqg_controller_test'

IMU_TOPIC_NAME = '/imu_topic'
ODOM_TOPIC_NAME = '/odom'
ERROR_TOPIC_NAME = '/error'
JOY_TOPIC_NAME = '/myteleop'


class LqgController(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.debug = False
        self.frame_id = 'base_link'
        self.QUEUE_SIZE = 10
        self.data_out_location_default = "/home/projects/ros2_ws/src/ucsd_robocar_hub2/ucsd_robocar_control2_pkg/data/"
        self.data_out_name_default = "test.csv"
        self.df = pd.DataFrame(columns = [\
            'time', \
            'joy_delta', \
            'joy_speed', \
            'lqg_delta', \
            'lqg_speed',\
            'lqg_e_cg',\
            'lqg_e_cg_dot',\
            'lqg_theta_e',\
            'lqg_theta_e_dot',\
            'lqg_e_cg_hat', \
            'lqg_e_cg_dot_hat', \
            'lqg_theta_e_hat', \
            'lqg_theta_e_dot_hat', \
            'yaw',\
            'yaw_rate',\
            'vx',\
            'vy'\
            ])

        # self.controller_thread = MutuallyExclusiveCallbackGroup()
        self.imu_thread = MutuallyExclusiveCallbackGroup()
        self.odom_thread = MutuallyExclusiveCallbackGroup()
        self.error_thread = MutuallyExclusiveCallbackGroup()
        self.joy_thread = MutuallyExclusiveCallbackGroup()

        # Actuator control
        self.drive_pub = self.create_publisher(AckermannDriveStamped, ACTUATOR_TOPIC_NAME, self.QUEUE_SIZE)
        self.drive_cmd = AckermannDriveStamped()

        ### Get sensor measurements ###
        #
        # Get IMU measurement
        self.imu_subscriber = self.create_subscription(Imu, IMU_TOPIC_NAME, self.imu_measurement, rclpy.qos.qos_profile_sensor_data, callback_group=self.imu_thread)
        self.imu_subscriber

        # Get Odometry measurements
        self.odom_subscriber = self.create_subscription(Odometry, ODOM_TOPIC_NAME, self.odom_measurement, self.QUEUE_SIZE, callback_group=self.odom_thread)
        self.odom_subscriber

        # Error subscriber
        self.error_subscriber = self.create_subscription(Float32MultiArray, ERROR_TOPIC_NAME, self.error_measurement, self.QUEUE_SIZE, callback_group=self.error_thread)
        self.error_subscriber

        # Get Joystick commands
        self.joy_subscriber = self.create_subscription(AckermannDriveStamped, JOY_TOPIC_NAME, self.set_joy_command, self.QUEUE_SIZE, callback_group=self.joy_thread)
        self.joy_subscriber

        self.start_time = time.time()
        self.current_time = self.get_clock().now().to_msg()

        # Sensor measurements
        self.v_min = 0.1
        self.yaw = 0
        self.yaw_rate = 0
        self.vx = self.v_min
        self.vy = 0
        self.joy_speed = 0
        self.joy_steering = 0 
        self.e_y_buffer = 0
        self.e_x_buffer = 0
        self.e_theta_buffer = 0

        # Sensor measurements Buffer
        # IMU
        self.yaw_imu_buffer = 0
        self.yaw_rate_imu_buffer = 0

        # VESC
        self.vx_vesc_buffer = 0.1
        self.vy_vesc_buffer = 0

        # JOY
        self.joy_speed_buffer = 0
        self.joy_steering_buffer = 0

        # Controller and State Estimate modules
        self.car_model = CarModel()
        self.lqr_calc = LQRDesign(self.car_model)
        self.kalman_calc = LinearKalmanFilter()
        self.ss_simulation = StateSpaceSimulation()
        self.P = np.diag([0, 0, 0, 0])
        self.Qo = np.diag([5.0E-3, 1.0E-2, 1.0E-6, 1.0E-6])
        self.Ro = np.diag([1.0E-3, 5.0E-1])
        self.x0 = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.state_measurement = self.x0
        self.state_est = self.x0
        self.sys = self.car_model.build_error_model(self.vx)

        # Calculated states
        self.e_y = 0  # cross-track error
        self.e_x = 0  # longitduinal error
        self.e_theta_m1 = 0  # previous heading error
        self.e_theta = 0  # heading error

        # Declare ROS parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('zero_speed', 0.0),
                ('max_speed', 5),
                ('min_speed', 0.1),
                ('error_threshold', 0.15),
                ('max_right_steering', 0.4),
                ('max_left_steering', -0.4),
                ('Ts', 0.05 ),
                ('data_out_location', self.data_out_location_default),
                ('data_out_name', self.data_out_name_default )
            ])
        self.zero_speed = self.get_parameter('zero_speed').value  # should be around 0
        self.max_speed = self.get_parameter('max_speed').value  # between [0,5] m/s
        self.min_speed = self.get_parameter('min_speed').value  # between [0,5] m/s 
        self.error_threshold = self.get_parameter('error_threshold').value  # between [0,1]
        self.max_right_steering = self.get_parameter('max_right_steering').value  # negative(max_left) 
        self.max_left_steering = self.get_parameter('max_left_steering').value  # between abs([0,0.436332]) radians (0-25degrees)
        self.Ts = self.get_parameter('Ts').value
        self.data_out_location = self.get_parameter('data_out_location').value
        self.data_out_name = self.get_parameter('data_out_name').value
        self.delta_normalization = max(abs(self.max_right_steering),abs(self.max_left_steering))

        self.data_out = self.data_out_location+self.data_out_name+".csv"

        self.get_logger().info(
            f'\n zero_speed: {self.zero_speed}'
            f'\n max_speed: {self.max_speed}'
            f'\n min_speed: {self.min_speed}'
            f'\n error_threshold: {self.error_threshold}'
            f'\n max_right_steering: {self.max_right_steering}'
            f'\n max_left_steering: {self.max_left_steering}'
            f'\n controller sample time: {self.Ts}'
            f'\n data_out file: {self.data_out}'
        )

        # Call controller
        self.create_timer(self.Ts, self.controller)
        self.create_timer(self.Ts, self.save_csv)

    def imu_measurement(self, imu_data):
        quaternion = (imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w)
        euler = euler_from_quaternion(quaternion)
        
        self.yaw_imu_buffer = euler[2]
        self.yaw_rate_imu_buffer = imu_data.angular_velocity.z
        # self.get_logger().info(f"Updating IMU: {(180 / math.pi) * self.yaw_imu_buffer}, {self.yaw_rate_imu_buffer}")

    def odom_measurement(self, odom_data):
        # car position
        self.x_vesc_buffer = odom_data.pose.pose.position.x
        self.y_vesc_buffer = odom_data.pose.pose.position.y

        # car linear velocity
        self.vx_vesc_buffer = odom_data.twist.twist.linear.x
        self.vy_vesc_buffer = odom_data.twist.twist.linear.y

        # car orientation
        quaternion = (\
            odom_data.pose.pose.orientation.x, \
            odom_data.pose.pose.orientation.y, \
            odom_data.pose.pose.orientation.z, \
            odom_data.pose.pose.orientation.w \
            )
        euler = euler_from_quaternion(quaternion)
        self.yaw_vesc_buffer = euler[2]

        # car angular velocity
        self.yaw_rate_vesc_buffer = odom_data.twist.twist.angular.z

    def error_measurement(self, error_data):
        error_data_check = np.array([error_data.data[0], error_data.data[1], error_data.data[2]])
        if not (np.isnan(error_data_check).any()):
            self.e_y_buffer = error_data.data[0]
            self.e_x_buffer = error_data.data[1]
            self.e_theta_buffer = error_data.data[2]

    def set_joy_command(self, joy_data):
        self.joy_speed_buffer = joy_data.drive.speed
        self.joy_steering_buffer = joy_data.drive.steering_angle

    def get_latest_measurements(self):
        # car orientation
        # self.yaw = float(np.mean([self.yaw_lidar_buffer, self.yaw_imu_buffer, self.yaw_vesc_buffer]))
        self.yaw = self.yaw_imu_buffer

        # car angular speed
        # self.yaw_rate = float(np.mean([self.yaw_rate_imu_buffer, self.yaw_rate_vesc_buffer]))
        self.yaw_rate = self.yaw_rate_imu_buffer
        
        # car linear speed
        self.vx = max(self.v_min, self.vx_vesc_buffer)
        self.vy = self.vy_vesc_buffer

        # error data from lidar
        self.e_y = self.e_y_buffer
        self.e_x = self.e_x_buffer
        self.e_theta_m1 = self.e_theta
        self.e_theta = self.e_theta_buffer

        # manual control
        self.joy_speed = self.joy_speed_buffer 
        self.joy_steering = self.joy_steering_buffer

        # time
        self.current_time = self.get_clock().now().to_msg()

        # update states
        self.state_measurement[0][0] = self.e_y
        self.state_measurement[1][0] = self.vy + self.vx * math.sin(self.e_theta)
        self.state_measurement[2][0] = self.e_theta
        self.state_measurement[3][0] = (self.e_theta - self.e_theta_m1) / self.Ts

    def update_gains(self):
        K = self.lqr_calc.compute_single_gain_sample(self.vx, self.sys)
        return K

    def controller(self):
        """
        Need:
        -pose data
        -path data
        -vx (measured longitudinal velocity)

        states:
        -ecg (cross-trackk error from center of gravity (cg))
            -state_measurement[0][0]
        -ecg_dot (cross-trackk error rate from cg)
            -state_measurement[1][0]
        -theta_e (heading error)
            -state_measurement[2][0]
        -theta_e_dot (heading error rate)
            -state_measurement[3][0]

        inputs:
        -delta (steering angle)

        published message:
        -steering_angle (radians)
        -speed (m/s)
        """
        
        # Update Car model LTV system --- A(Vx)
        self.sys = self.car_model.build_error_model(self.vx, 2)

        # get updated gains
        K = self.update_gains()

        # Steering LQR
        self.delta_raw = -np.dot(K[0], self.state_est).flat[0]
        delta = self.clamp(self.delta_raw, self.max_right_steering, self.max_left_steering)

        # Throttle gain scheduling
        # normalized_delta = delta / self.delta_normalization
        # self.inf_throttle = self.min_speed - (self.min_speed - self.max_speed) / (1 - self.error_threshold)
        # speed_raw = ((self.min_speed - self.max_speed) / (1 - self.error_threshold)) * abs(normalized_delta) + self.inf_throttle
        speed_raw = 0.3
        speed = self.clamp(speed_raw, self.max_speed, self.min_speed)

        # Get Current Measurement
        self.y_sim = self.car_model.calc_output(self.state_measurement)

        # Get optimal state estimates
        self.state_est, self.P = self.kalman_calc.lkf(self.sys, self.state_est, self.joy_steering, self.y_sim, self.P, self.Qo, self.Ro)
        
        if self.debug:
            self.get_logger().info(
                f'\n e_cg: {self.state_measurement[0][0]}'
                f'\n e_cg_dot: {self.state_measurement[1][0]}'
                f'\n theta_e: {self.state_measurement[2][0]}'
                f'\n theta_e_dot: {self.state_measurement[3][0]}'
                f'\n yaw: {self.yaw}'
                f'\n yaw_rate: {self.yaw_rate}'
                f'\n vx: {self.vx}'
                f'\n vy: {self.vy}'
                f'\n joy_speed: {self.joy_speed}'
                f'\n joy_steering: {self.joy_steering}'
                f'\n y: {self.y}'
                f'\n state_est: {self.state_est}'
            )

        # Publish values
        try:
            # publish drive control signal
            self.drive_cmd.header.stamp = self.current_time
            self.drive_cmd.header.frame_id = self.frame_id
            self.drive_cmd.drive.speed = speed
            self.drive_cmd.drive.steering_angle = delta
            self.drive_pub.publish(self.drive_cmd)

        except KeyboardInterrupt:
            self.drive_cmd.header.stamp = self.current_time
            self.drive_cmd.header.frame_id = self.frame_id
            self.drive_cmd.drive.speed = self.zero_speed
            self.drive_cmd.drive.steering_angle = 0
            self.drive_pub.publish(self.drive_cmd)
            
        # Get new sensor measurements
        self.get_latest_measurements()

        # write out
        self.compare_manual_and_lqr()
        
    def compare_manual_and_lqr(self):
        self.df = pd.concat([self.df, pd.DataFrame.from_records([{ \
            'time': float(round((time.time() - self.start_time),3)), \
            'joy_delta': float(round(self.joy_steering,3)), \
            'joy_speed': float(round(self.joy_speed,3)), \
            'lqg_delta': float(round(self.delta_raw,3)), \
            'lqg_speed': float(round(self.drive_cmd.drive.speed,3)), \
            'lqg_e_cg': float(round(self.state_measurement[0][0],3)), \
            'lqg_e_cg_dot': float(round(self.state_measurement[1][0],3)), \
            'lqg_theta_e': float(round(self.state_measurement[2][0],3)), \
            'lqg_theta_e_dot': float(round(self.state_measurement[3][0],3)), \
            'lqg_e_cg_hat': float(round(self.state_est[0][0],3)), \
            'lqg_e_cg_dot_hat': float(round(self.state_est[1][0],3)), \
            'lqg_theta_e_hat': float(round(self.state_est[2][0],3)), \
            'lqg_theta_e_dot_hat': float(round(self.state_est[3][0],3)), \
            'yaw': float(round(self.yaw,3)), \
            'yaw_rate': float(round(self.yaw_rate,3)), \
            'vx': float(round(self.vx,3)), \
            'vy': float(round(self.vy,3)) \
            }])])

    def save_csv(self):
        self.df.to_csv(self.data_out, index = False)

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
        executor = MultiThreadedExecutor(num_threads=5)
        executor.add_node(lqg_publisher)
        try:
            executor.spin()
        finally:
            lqg_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
            lqg_publisher.drive_cmd.header.stamp = lqg_publisher.current_time
            lqg_publisher.drive_cmd.header.frame_id = lqg_publisher.frame_id
            lqg_publisher.drive_cmd.drive.speed = 0.0
            lqg_publisher.drive_cmd.drive.steering_angle = 0.0
            lqg_publisher.drive_pub.publish(lqg_publisher.drive_cmd)
            time.sleep(1)
            lqg_publisher.save_csv()
            lqg_publisher.get_logger().info(f'Saved data to: {lqg_publisher.data_out}.')
            executor.shutdown()
            lqg_publisher.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
