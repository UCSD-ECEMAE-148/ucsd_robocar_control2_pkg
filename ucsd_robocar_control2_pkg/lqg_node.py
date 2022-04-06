import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
from sensor_msgs.msg import IMU
from .controller_submodule.lqr_calculator import LQRDesign
from .controller_submodule.car_model import CarModel
from .filter_submodule.linear_kalman_calculator import LinearKalmanFilter
import time
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
        self.Ts = 100 # imu sample frequency (Hz)
        self.vx = 0
        self.vy = 0

        # Get GPS/Lidar measurements
        self.pose_subscriber = self.create_subscription(Pose, POSE_TOPIC_NAME, self.pose_measurement, 10)
        self.pose_subscriber
        self.path_subscriber = self.create_subscription(Path, PATH_TOPIC_NAME, self.set_path, 10)
        self.path_subscriber

        # Get road marker error measurements from camera
        self.pose_error_subscriber = self.create_subscription(Float32MultiArray, ERROR_TOPIC_NAME, self.camera_measurment, 10)
        self.pose_error_subscriber

        # Controller modules
        self.car_model = CarModel()
        self.lqr_calc = LQRDesign()
        self.kalman_calc = LinearKalmanFilter()
        self.P0 = np.diag([1, 1, 1, 1])
        self.Qo = np.diag([1, 1, 1, 1])
        self.Ro = [0.1]
        self.x0 = np.array([0, 0, 0, 0])
        self.state = self.x0
        self.u = 0

        self.create_timer(self.Ts, self.controller)

        # Default actuator values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('error_threshold', 0.15),
                ('zero_throttle',0.0),
                ('max_throttle', 0.2),
                ('min_throttle', 0.1),
                ('max_right_steering', 1.0),
                ('max_left_steering', -1.0)
            ])
        self.error_threshold = self.get_parameter('error_threshold').value # between [0,1]
        self.zero_throttle = self.get_parameter('zero_throttle').value # between [-1,1] but should be around 0
        self.max_throttle = self.get_parameter('max_throttle').value # between [-1,1]
        self.min_throttle = self.get_parameter('min_throttle').value # between [-1,1]
        self.max_right_steering = self.get_parameter('max_right_steering').value # between [-1,1]
        self.max_left_steering = self.get_parameter('max_left_steering').value # between [-1,1]

        # initializing control
        self.Ts = float(1/20)
        
        self.get_logger().info(
            f'\nerror_threshold: {self.error_threshold}'
            f'\nzero_throttle: {self.zero_throttle}'
            f'\nmax_throttle: {self.max_throttle}'
            f'\nmin_throttle: {self.min_throttle}'
            f'\nmax_right_steering: {self.max_right_steering}'
            f'\nmax_left_steering: {self.max_left_steering}'
        )

    def imu_measurement(self, imu_data):
        quaternion = (imu_data.orientation.x, imu_data.orientation.y, imu_data.orientation.z, imu_data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        
        # orientation
        self.roll_imu = euler[0]
        self.pitch_imu = euler[1]
        self.yaw_imu = euler[2]

        # angular velocity
        self.roll_rate = imu_data.angular_velocity.x
        self.pitch_rate = imu_data.angular_velocity.y
        self.yaw_rate = imu_data.angular_velocity.z

        # linear acceleration
        self.ax = imu_data.linear_acceleration.x
        self.ay = imu_data.linear_acceleration.y
        self.az = imu_data.linear_acceleration.z

        # linear velocity
        self.vx = self.vx + (ax * self.Ts) 
        self.vy = self.vy + (ay * self.Ts)

    def pose_measurement(self, pose_data):
        quaternion = (pose_data.orientation.x, pose_data.orientation.y, pose_data.orientation.z, pose_data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        
        # car orientation
        self.roll_pose_measurement = euler[0]
        self.pitch_pose_measurement = euler[1]
        self.yaw_pose_measurement = euler[2]

        # car coordinates
        self.x = pose_data.position.x
        self.y = pose_data.position.y
        self.z = pose_data.position.z

    def set_path(self, path_data):
        quaternion = (path_data.orientation.x, path_data.orientation.y, path_data.orientation.z, path_data.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        
        # path orientation
        self.roll_path= euler[0]
        self.pitch_path= euler[1]
        self.yaw_path= euler[2]

        # path coordinates
        self.x_path = pose_data.position.x
        self.y_path = pose_data.position.y
        self.z_path = pose_data.position.z

    def camera_measurment(self, error_data):
        """
        Need:
        -pose data and path data to calculate errors 

        ecg: cross-trackk error from center of gravity (cg)
        ecg_dot: cross-trackk error rate from cg
        theta_e: heading error
        theta_e_dot: heading error rate
        """
        self.ecg = error_data.data[0] # cross-track error (pose_error_y * delta_x_path - pose_error_x * delta_y_path) / (delta_x_path^2 + delta_y_path^2)
        self.ecg_dot = error_data.data[1] # ecg_dot = vy + vx * sin(theta_error);
        self.theta_e = error_data.data[2] # theta_e = path_angle - car_yaw_angle
        self.theta_e_dot = error_data.data[3] # theta_e_dot = (theta_e_k - theta_e_km1) / self.Ts # theta_e_k = heading error at sample k AND theta_e_km1 = heading error at sample k - 1


    def controller(self):
        # Throttle gain scheduling (function of error)
        self.inf_throttle = self.min_throttle - (self.min_throttle - self.max_throttle) / (1 - self.error_threshold)
        throttle_float_raw = ((self.min_throttle - self.max_throttle)  / (1 - self.error_threshold)) * abs(self.ek) + self.inf_throttle
        throttle_float = self.clamp(throttle_float_raw, self.max_throttle, self.min_throttle)

        # Update Car model 
        sys = self.car_model.build_error_model(self.vx)
        self.y = self.car_model.calc_output(self.state)

        # Get gains
        K = self.lqr_calc.compute_gain_constant_speed()
        K1 = K.flat[0]
        K2 = K.flat[1]
        K3 = K.flat[2]
        K4 = K.flat[3]
        
        # Get optimal state estimates
        state_est = self.kalman_calc.lkf(sys, self.x0, self.u, self.y, self.P0, self.Qo, self.Ro).flat
        ecg = state_est[0]
        ecg_dot = state_est[1]
        theta_e = state_est[2]
        theta_e_dot = state_est[3]
        
        steering_float_raw = K1 * ecg + K2 * ecg_dot + K3 * theta_e  + K4 * theta_e_dot
        steering_float = self.clamp(steering_float_raw, self.max_right_steering, self.max_left_steering)

        # Publish values
        try:
            # publish control signals
            self.twist_cmd.angular.z = steering_float
            self.twist_cmd.linear.x = throttle_float
            self.twist_publisher.publish(self.twist_cmd)

        except KeyboardInterrupt:
            self.twist_cmd.linear.x = self.zero_throttle
            self.twist_publisher.publish(self.twist_cmd)



    def clamp(self, value, upper_bound, lower_bound=None):
        if lower_bound==None:
            lower_bound = -upper_bound # making lower bound symmetric about zero
        if value < lower_bound:
            value_c = lower_bound
        elif value > upper_bound:
            value_c = upper_bound
        else:
            value_c = value
        return value_c 


def main(args=None):
    rclpy.init(args=args)
    lqr_publisher = LqgController()
    try:
        rclpy.spin(lqr_publisher)
        lqr_publisher.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        lqr_publisher.get_logger().info(f'Shutting down {NODE_NAME}...')
        lqr_publisher.twist_cmd.linear.x = lqr_publisher.zero_throttle
        lqr_publisher.twist_publisher.publish(lqr_publisher.twist_cmd)
        time.sleep(1)
        lqr_publisher.destroy_node()
        rclpy.shutdown()
        lqr_publisher.get_logger().info(f'{NODE_NAME} shut down successfully.')


if __name__ == '__main__':
    main()

