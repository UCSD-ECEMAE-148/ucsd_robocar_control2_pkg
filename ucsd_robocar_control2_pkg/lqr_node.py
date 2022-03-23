import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
import time
import os

NODE_NAME = 'lqr_node'
ACTUATOR_TOPIC_NAME = '/cmd_vel'

POSE_TOPIC_NAME = '/pose'
PATH_TOPIC_NAME = '/path'
ERROR_TOPIC_NAME = '/path_error'

class LqrController(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.twist_publisher = self.create_publisher(Twist, ACTUATOR_TOPIC_NAME, 10)
        self.twist_cmd = Twist()

        # One or the other...
        self.pose_subscriber = self.create_subscription(Pose, POSE_TOPIC_NAME, self.set_pose, 10)
        self.pose_subscriber
        self.path_subscriber = self.create_subscription(Path, PATH_TOPIC_NAME, self.set_path, 10)
        self.path_subscriber
        # OR
        self.pose_error_subscriber = self.create_subscription(Float32MultiArray, ERROR_TOPIC_NAME, self.controller, 10)
        self.pose_error_subscriber
        


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

    def set_pose(self,data):
        pass

    def set_path(self,data):
        pass

    def controller(self, data):
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
        # setting up LQR control

        # OR
        self.ecg = data.data[0]
        self.ecg_dot = data.data[1]
        self.theta_e = data.data[2]
        self.theta_e_dot = data.data[3]

        # Throttle gain scheduling (function of error)
        self.inf_throttle = self.min_throttle - (self.min_throttle - self.max_throttle) / (1 - self.error_threshold)
        throttle_float_raw = ((self.min_throttle - self.max_throttle)  / (1 - self.error_threshold)) * abs(self.ek) + self.inf_throttle
        throttle_float = self.clamp(throttle_float_raw, self.max_throttle, self.min_throttle)

        # Steering LQR (TODO: add functions to calculate parameters below)
        steering_float_raw = = self.K1 * self.ecg + self.K2 * self.ecg_dot + self.K3 * self.theta_e  + self.K4 * self.theta_e_dot
        # OR

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
    lqr_publisher = LqrController()
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
