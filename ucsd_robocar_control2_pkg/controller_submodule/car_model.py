from control import *
from control.matlab import *  # MATLAB-like functions
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import yaml


class CarModel:
    def __init__(self, car_parameter_input_path=None):
        self.Ts = 0.01
        self.m = 630  # FIXME total mass (kg)
        self.mf = self.m * 0.42  # FIXME mass on front axel
        self.mr = self.m * 0.58  # FIXME mass on rear axel
        self.L = 2.9718  # FIXME wheel base (meters)
        self.cf = 2 * 8E4  # FIXME front tire cornering stiffness
        self.cr = 2 * 8E4  # FIXME rear tire cornering stiffness
        self.Lf = self.L * (1 - self.mf / self.m)  # distance from CG to front axel
        self.Lr = self.L * (1 - self.mr / self.m)  # distance from CG to rear axel
        self.Iz = self.Lf * self.Lr * (self.mf + self.mr)  # moment of inertia
        self.sysd = 0
        self.car_parameter_input_path = car_parameter_input_path
        if self.car_parameter_input_path is not None:
            self.update_parameters()

    def update_parameters(self):
        with open(self.car_parameter_input_path, "r") as car_parameter_file:
            car_inputs = yaml.load(car_parameter_file, Loader=yaml.FullLoader)
            self.car_parameter_input_dictionary = eval(car_inputs)
            for key in self.car_parameter_input_dictionary:
                value = self.car_parameter_input_dictionary[key]
                setattr(self, key, value)

    def build_error_model(self, Vx):
        a11 = 0
        a12 = 1
        a13 = 0
        a14 = 0
        a21 = 0
        a22 = -(self.cf + self.cr) / (self.m * Vx)
        a23 = (self.cf + self.cr) / self.m
        a24 = (self.Lr * self.cr - self.Lf * self.cf) / (self.m * Vx)
        a31 = 0
        a32 = 0
        a33 = 0
        a34 = 1
        a41 = 0
        a42 = (self.Lr * self.cr - self.Lf * self.cf) / (self.Iz * Vx)
        a43 = (self.Lf * self.cf - self.Lr * self.cr) / self.Iz
        a44 = -((self.Lr ** 2) * self.cr + (self.Lf ** 2) * self.cf) / (self.Iz * Vx)

        A = np.matrix(
            [[a11, a12, a13, a14],
             [a21, a22, a23, a24],
             [a31, a32, a33, a34],
             [a41, a42, a43, a44]]
        )

        # Input matrix
        B = np.matrix(
            [[0],
             [self.cf / self.m],
             [0],
             [self.Lf * self.cf / self.Iz]]
        )

        # Output matrix
        C = np.matrix(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             ])

        # Feed-Forward matrix
        D = np.matrix(
            [[0],
             [0],
             [0],
             [0]]
        )

        sys = ss(A, B, C, D)
        self.sysd = c2d(sys, self.Ts, method='zoh')
        return self.sysd


def build_model_example():
    V_x = 3
    my_car_model = CarModel()
    my_sys = my_car_model.build_error_model(V_x)
    [A, B, C, D] = ssdata(my_sys)
    print(f"A: {A}"
          f"\nB: {B}"
          f"\nC: {C}"
          f"\nD: {D}")


if __name__ == '__main__':
    build_model_example()
