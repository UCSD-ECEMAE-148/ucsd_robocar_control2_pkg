from control import *
from control.matlab import *  # MATLAB-like functions
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import yaml
from .controller_submodule.car_model import *


class LQRDesign:
    def __init__(self, car_parameter_input_path):
        self.Ts = 0.01
        self.sysd = 0
        self.car_parameter_input_path = car_parameter_input_path
        self.lqr_car = CarModel()
        self.sysd = 0

    def build_system(self, Vx):
        self.sysd = self.lqr_car.build_error_model(Vx)

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

    def curve_fit_power_func(self, x_input, a, b, c):
        result = a * np.power(x_input, b) + c
        return result

    def my_curve_fit(self, x_input, y_output):
        popt, _ = curve_fit(self.curve_fit_power_func, x_input, y_output, maxfev=100000)
        a, b, c = popt
        y_fit = self.curve_fit_power_func(x_input, a, b, c)
        return y_fit
        

def plotting_example():
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


if __name__ == '__main__':
    plotting_example()

