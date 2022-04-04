from control import *
from control.matlab import *  # MATLAB-like functions
import numpy as np
from build_car_model import CarModel


class StateSpaceSimulation:
    def __init__(self):
        self.sample_size = 0
        self.x = np.zeros([1, 1], dtype=float)
        self.y = np.zeros([1, 1], dtype=float)
        self.lqr_car = CarModel()
        self.sysd = 0

    def build_system(self, Vx):
        self.sysd = self.lqr_car.build_error_model(Vx)

    def ss_simulation(self, A, B, C, D, G, x0, u, d, v):
        self.sample_size = u.size
        a_num_rows, a_num_cols = A.shape
        d_num_rows, d_num_cols = D.shape
        self.x = np.zeros([a_num_rows, self.sample_size], dtype=float)
        self.y = np.zeros([d_num_rows, self.sample_size], dtype=float)
        self.x[:, 0] = x0.transpose()
        for k in range(0, self.sample_size):
            self.y[:, k] = np.add(np.dot(C, self.x[:, k]), np.add(np.dot(D, u[k]), v[:, k]))
            self.x[:, k + 1] = np.add(np.dot(A, self.x[:, k]), np.add(np.dot(B, u[k]), np.dot(G, d[k])))
        return self.x, self.y


def main():
    v = 5 # m/s
    my_sim = StateSpaceSimulation()
    my_sys = my_sim.build_system(v)
    [A, B, C, D] = ssdata(my_sys)
    a_num_rows, a_num_cols = A.shape
    d_num_rows, d_num_cols = D.shape
    
    # G = np.zeros([a_num_rows, self.sample_size], dtype=float)
    # x0 = [1, 1, 1, 1]
    # u = 
    # d = 
    # v = 
    # ss_simulation(self, A, B, C, D, G, x0, u, d, v)


if __name__ == '__main__':
    main()
