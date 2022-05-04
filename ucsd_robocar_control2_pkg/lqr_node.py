import time
import os
import rclpy
from rclpy.node import Node 
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu
from .controller_submodule.lqr_calculator import LQRDesign
from .controller_submodule.car_model import CarModel
import numpy as np

NODE_NAME = 'lqr_node'
ACTUATOR_TOPIC_NAME = '/cmd_vel'

POSE_TOPIC_NAME = '/amcl_pose'
PATH_TOPIC_NAME = '/global_trajectory'
IMU_TOPIC_NAME = '/razor/imu'
ODOM_TOPIC_NAME = '/odom'

class LqrController(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
         
        self.controller_thread = MutuallyExclusiveCallbackGroup()
        self.path_thread = MutuallyExclusiveCallbackGroup()
        self.odom_thread = MutuallyExclusiveCallbackGroup()
        self.pose_thread = MutuallyExclusiveCallbackGroup()

        self.twist_publisher = self.create_publisher(Twist, ACTUATOR_TOPIC_NAME, 10)
        self.twist_cmd = Twist()

        ### Get sensor measurements ###
        #
        # Get GPS/Lidar measurements
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, POSE_TOPIC_NAME, self.pose_measurement, callback_group=self.pose_thread, 10)
        self.pose_subscriber

        # Get Odometry measurements
        self.odom_subscriber = self.create_subscription(Odometry, ODOM_TOPIC_NAME, self.odom_measurement, callback_group=self.odom_thread, 10)
        self.odom_subscriber

        # Get Reference Trajectory
        self.path_subscriber = self.create_subscription(Path, PATH_TOPIC_NAME, self.set_path, callback_group=self.path_thread, 10)
        self.path_subscriber

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
        self.vx = 0.1
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.az = 0


        # Controller modules
        self.car_model = CarModel()
        self.lqr_calc = LQRDesign(self.car_model)
        self.x0 = np.array([[0.0], [0.0], [0.0], [0.0]])
        self.state_measurement = self.x0
        self.sys = self.car_model.build_error_model(self.vx)

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
        self.theta_e_dot = 0  # heading error yaw_rate

        # Default actuator values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('k1_gain', 1.0),
                ('k2_gain', 1.0),
                ('k3_gain', 1.0),
                ('k4_gain', 1.0),
                ('k1_coeff', [1.0, 1.0, 1.0]),
                ('k2_coeff', [1.0, 1.0, 1.0]),
                ('k3_coeff', [1.0, 1.0, 1.0]),
                ('k4_coeff', [1.0, 1.0, 1.0]),
                ('error_threshold', 0.15),
                ('zero_throttle', 0.0),
                ('max_throttle', 0.2),
                ('min_throttle', 0.1),
                ('max_right_steering', 1.0),
                ('max_left_steering', -1.0)
            ])

        self.k1_gain=self.get_parameter('k1_gain').value
        self.k2_gain=self.get_parameter('k2_gain').value
        self.k3_gain=self.get_parameter('k3_gain').value
        self.k4_gain=self.get_parameter('k4_gain').value
        self.k1_coeff=self.get_parameter('k1_coeff').value
        self.k2_coeff=self.get_parameter('k2_coeff').value
        self.k3_coeff=self.get_parameter('k3_coeff').value
        self.k4_coeff=self.get_parameter('k4_coeff').value
        self.error_threshold=self.get_parameter('error_threshold').value  # between [0,1]
        self.zero_throttle=self.get_parameter('zero_throttle').value  # between [-1,1] but should be around 0
        self.max_throttle=self.get_parameter('max_throttle').value  # between [-1,1]
        self.min_throttle=self.get_parameter('min_throttle').value  # between [-1,1]
        self.max_right_steering=self.get_parameter('max_right_steering').value  # between [-1,1]
        self.max_left_steering=self.get_parameter('max_left_steering').value  # between [-1,1]

        self.get_logger().info(
            f'\nk1_gain: {self.k1_gain}'
            f'\nk2_gain: {self.k2_gain}'
            f'\nk3_gain: {self.k3_gain}'
            f'\nk4_gain: {self.k4_gain}'
            f'\nk1_coeff: {self.k1_coeff}'
            f'\nk2_coeff: {self.k2_coeff}'
            f'\nk3_coeff: {self.k3_coeff}'
            f'\nk4_coeff: {self.k4_coeff}'
            f'\nerror_threshold: {self.error_threshold}'
            f'\nzero_throttle: {self.zero_throttle}'
            f'\nmax_throttle: {self.max_throttle}'
            f'\nmin_throttle: {self.min_throttle}'
            f'\nmax_right_steering: {self.max_right_steering}'
            f'\nmax_left_steering: {self.max_left_steering}'
        )

        # Call controller
        self.Ts = 1/100  # contoller publish frequency (Hz)
        self.create_timer(self.Ts, self.controller)

    def odom_measurement(self, odom_data):
        # position
        self.x = odom_data.pose.pose.position.x
        self.y = odom_data.pose.pose.position.y
        self.z = odom_data.pose.pose.position.z

        # FIXME: confirm coordinate axes
        # orientation
        quaternion = (odom_data.orientation.x, odom_data.orientation.y, odom_data.orientation.z, odom_data.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]

        # velocity
        self.vx = odom_data.twist.twist.linear.x
        self.vy = odom_data.twist.twist.linear.y

    def pose_measurement(self, pose_data):
        self.get_logger().info("Updating POSE")

        # TODO: what is frequency of data coming in?
        # FIXME: confirm coordinate axes
        # car coordinates
        self.x = pose_data.position.x
        self.y = pose_data.position.y
        self.z = pose_data.position.z

    def set_path(self, path_data):
        self.get_logger().info("Updating PATH")

        # TODO: Currently not working with Lidar Nav
        # FIXME: confirm coordinate axes
        # path orientation 
        # quaternion = (path_data.poses[0].pose.orientation.x, path_data.poses[0].pose.orientation.y,
        #               path_data.poses[0].pose.orientation.z, path_data.poses[0].pose.orientation.w)
        # euler = euler_from_quaternion(quaternion)
        # self.roll_path = euler[0]
        # self.pitch_path = euler[1]
        # self.yaw_path = euler[2]

        # path coordinates (GLOBAL)
        self.x_path = path_data.poses[0].pose.position.x
        self.y_path = path_data.poses[0].pose.position.y
        self.z_path = path_data.poses[0].pose.position.z

    def calc_cross_track_error(self):
        efa_x = self.x_path - self.x
        efa_y = self.y_path - self.y
        efa_mag = np.power(np.power(efa_x,2) + np.power(efa_y, 2), 0.5);
        efa_mag1, efa_mag2 = np.partition(efa_mag, 1)[0:2]
        efa_mag1_index = np.where(efa_mag == efa_mag1)
        efa_mag2_index = np.where(efa_mag == efa_mag2)
        Px1 = self.x_path[efa_mag1_index]
        Px2 = self.x_path[efa_mag2_index]
        Py1 = self.y_path[efa_mag1_index]
        Py2 = self.y_path[efa_mag2_index]
        delta_x = Px2 - Px1
        delta_y = Py2 - Py1
        R_x = self.x - Px1
        R_y = self.y - Py1
        r_2 = np.power(delta_x, 2) + np.power(delta_y, 2)
        e_cg = (R_y * delta_x - R_x * delta_y) / r_2
        return e_cg, e_cg_index

    def update_gains(self):
        # self.get_logger().info("Updating GAINS")
        # K_mat=[]
        # # put all coeff for each gain function into matrix with dim: 4x3
        # coeff_mat=[self.k1_coeff, self.k2_coeff, self.k3_coeff, self.k4_coeff]
        # for coeff in coeff_mat:
        #     K = self.calc_gain_power_function(coeff)
        #     K_mat.append(K)
        # self.K1=K_mat[0]
        # self.K2=K_mat[1]
        # self.K3=K_mat[2]
        # self.K4=K_mat[3]
        K = self.lqr_calc.compute_single_gain_sample(self.sys)
        return K

    def update_states(self):
        self.get_logger().info("Updating STATES")
        delta_x_path = self.x_path[1] - self.x_path[0]
        delta_y_path = self.y_path[1] - self.y_path[0]
        pose_error_x = self.x - self.x_path[0]
        pose_error_y = self.y - self.y_path[0]
        theta_e_km1 = self.state_measurement[0][2]
        e_cg, e_cg_index = self.calc_cross_track_error()
        theta_e_k = self.theta_p[e_cg_index] - self.yaw_imu  # Path needs to be in reference with car not map (local path)
        self.state_measurement[0][0] = e_cg
        self.state_measurement[0][1] = self.vy + self.vx * math.sin(theta_e_k)
        self.state_measurement[0][2] = theta_e_k
        self.state_measurement[0][3] = (theta_e_k - theta_e_km1) / self.Ts

    def controller(self):
        """
        Need:
        -pose data and path data to calculate errors
        OR
        -previously calculated errors

        ecg: cross-trackk error from center of gravity (cg)
        ecg_dot: cross-trackk error rate from cg
        theta_e: heading error
        theta_e_dot: heading error rate
        """

        # Update Car model LTV system --- A(Vx)
        self.sys = self.car_model.build_error_model(self.vx)

        # get updated gains
        K = self.update_gains()

        # Steering LQR
        steering_float_raw = -np.dot(K[0], self.state_measurement).flat[0]

        # TODO: function of cross-track error (e_cg) or heading error (theta_e)??
        # Throttle gain scheduling
        tracking_error = self.state_measurement[0][0]
        self.inf_throttle = self.min_throttle - (self.min_throttle - self.max_throttle) / (1 - self.error_threshold)
        throttle_float_raw = ((self.min_throttle - self.max_throttle) / (1 - self.error_threshold)) * abs(tracking_error) + self.inf_throttle

        # Clamp control inputs
        # FIXME: need to convert to radians and m/s respectively 
        steering_float = self.clamp(steering_float_raw, self.max_right_steering, self.max_left_steering)
        throttle_float = self.clamp(throttle_float_raw, self.max_throttle, self.min_throttle)

        # Publish values
        try:
            # publish control signals
            self.twist_cmd.angular.z=steering_float
            self.twist_cmd.linear.x=throttle_float
            self.twist_publisher.publish(self.twist_cmd)

        except KeyboardInterrupt:
            self.twist_cmd.linear.x=self.zero_throttle
            self.twist_publisher.publish(self.twist_cmd)

    def clamp(self, value, upper_bound, lower_bound=None):
        if lower_bound == None:
            lower_bound=-upper_bound  # making lower bound symmetric about zero
        if value < lower_bound:
            value_c=lower_bound
        elif value > upper_bound:
            value_c=upper_bound
        else:
            value_c=value
        return value_c


def main(args=None):
    rclpy.init(args=args)
    lqr_publisher=LqrController()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(lqr_publisher)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            lqr_publisher.destroy_node()
    except KeyboardInterrupt:
        lqr_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        lqr_publisher.twist_cmd.linear.x=lqr_publisher.zero_throttle
        lqr_publisher.twist_publisher.publish(lqr_publisher.twist_cmd)
        time.sleep(1)
        lqr_publisher.destroy_node()
        rclpy.shutdown()
        lqr_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()
