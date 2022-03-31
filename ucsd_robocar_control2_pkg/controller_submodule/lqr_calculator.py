from control import *
from control.matlab import *  # MATLAB-like functions
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import yaml


class LQRDesign:
    def __init__(self, car_parameter_input_path):
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
        self.update_parameters()
    
    def update_parameters(self):
        with open(self.car_parameter_input_path, "r") as car_parameter_file:
            car_inputs = yaml.load(car_parameter_file, Loader=yaml.FullLoader)
            self.car_parameter_input_dictionary = eval(car_inputs)
            for key in self.car_parameter_input_dictionary:
                value = self.car_parameter_input_dictionary[key]
                setattr(self, key, value)

    def build_model(self, Vx):
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

    def compute_q(self):
        Q = np.diag([2.0, 0.149999998734902, 1, 1]) # FIXME: update to vary as function of Vx
        return Q

    def compute_lqr_constant_speed(self):
        Q = self.compute_q()
        R = np.diag([0.001])
        K, S, E = lqr(self.sysd, Q * self.Ts, R / self.Ts)
        return K

    def compute_lqr_varying_speed(self, vx_vec):
        k_mat = np.empty((0,4))
        for vx in vx_vec:
            self.build_model(vx)
            k = self.compute_lqr_constant_speed()
            k_mat = np.append(k_mat, k, axis=0)
        return k_mat

    def curve_fit_objective(self, x_input, a, b, c):
        result = a * np.power(x_input, b) + c
        return result

    def my_curve_fit(self, x_input, y_output):
        popt, _ = curve_fit(self.curve_fit_objective, x_input, y_output, maxfev=100000)
        a, b, c = popt
        y_fit = self.curve_fit_objective(x_input, a, b, c)
        return y_fit


if __name__ == '__main__':
    num_sims = 20
    V_max = 80
    V_min = 3
    Vx_vec = linspace(V_min, V_max, num_sims)
    my_lqr = LQRDesign()
    K_mat = my_lqr.compute_lqr_varying_speed(Vx_vec)
    K_mat_shape = K_mat.shape
    try:
        for K in range(0,K_mat_shape[1]):
            plt.subplot(2, 2, K+1)
            plt.xlabel("Velocity (m/s)")
            plt.ylabel(f"K{K+1} Gain")
            k_t = K_mat[:, K].flat[:]
            k_t1 = np.squeeze(k_t[0])
            k_flat = list(np.concatenate(K_mat[:, K]).flat)
            # print(k_flat)
            plt.scatter(Vx_vec, k_flat)
            K_fit = my_lqr.my_curve_fit(Vx_vec, k_flat)
            plt.plot(Vx_vec, K_fit)
    except RuntimeError:
        pass
    plt.show()
