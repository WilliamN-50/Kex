import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class DifferentialEquation:
    def __init__(self, t_0, t_end, y_0):
        self.t_0 = t_0
        self.t_end = t_end
        self.y_0 = y_0
        self.num_y = len(y_0)

    def func(self, x, y):
        pass

    def integrate(self, method="RK45", t_points=None, out_file=None, save_to_file=False):
        solution = solve_ivp(self.func, [self.t_0, self.t_end], self.y_0, method=method, t_eval=t_points)
        rows = solution.t.shape[0]
        out_data = np.zeros((rows, 1+self.num_y))
        out_data[:, 0] = solution.t.T
        out_data[:, 1:] = solution.y.T
        if save_to_file:
            np.save(out_file, out_data)
        else:
            return out_data

    def reshape_data(self, in_data, out_file=None, save_to_file=False):
        """
        ____________________________
        Build a new data structure by combining data points from in_file.
        ____________________________
        """
        rows = in_data.shape[0]
        out_data = np.zeros((self.num_y, rows*(rows-1)//2, 5))
        n = 0
        for i in range(rows):
            for j in range(i+1, rows):
                for k in range(self.num_y):
                    out_data[k][n][0] = in_data[i][0]
                    out_data[k][n][1] = in_data[j][0]
                    out_data[k][n][2] = in_data[j][0] - in_data[i][0]
                    out_data[k][n][3] = in_data[i][1 + k]
                    out_data[k][n][4] = in_data[j][1 + k]
                n += 1
        if save_to_file:
            np.save(out_file, out_data)
        else:
            return out_data


class Diff_eq_0(DifferentialEquation):
    def func(self, t, y):
        return -y


class Diff_eq_1(DifferentialEquation):
    def func(self, x, y):
        return [y[0] - y[0]*y[1], -y[1] + y[0]*y[1]]


class Diff_eq_2(DifferentialEquation):
    def func(self, t, y):
        return 3/2 * y/(t+1) + np.sqrt(t+1)


def main():
    d_e0 = Diff_eq_1(t_0=0, t_end=10, y_0=[1, 2])
    data = d_e0.integrate(t_points=np.arange(0, 10, 0.1))
    reshaped_data = d_e0.reshape_data(data)
    print(data)
    print(reshaped_data)





if __name__ == '__main__':
    main()