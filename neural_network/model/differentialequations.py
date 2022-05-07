import abc
import numpy as np
from scipy.integrate import solve_ivp


class DifferentialEquation(metaclass=abc.ABCMeta):
    """
    ____________________________
    The DifferentialEquation class.
    Generates data for NeuralNetworkModel.
    ____________________________
    """
    def __init__(self, t_0, t_end, y_0):
        self.t_0 = t_0
        self.t_end = t_end
        self.y_0 = y_0
        self.num_y = len(y_0)

    @staticmethod
    @abc.abstractmethod
    def func(x, y):
        pass

    def integrate(self, method="RK45", rel_tol=1e-6, t_points=None, noise_level=0.0, out_file=None,
                  save_to_file=False):
        """
        ____________________________
        Integrates the differential equation.
        ____________________________
        """
        solution = solve_ivp(self.func, [self.t_0, self.t_end], self.y_0, method=method, t_eval=t_points, rtol=rel_tol)
        rows = len(solution.t)
        out_data = np.empty((rows, 1 + self.num_y))
        out_data[:, 0] = solution.t.T
        if noise_level > 0:
            noise = np.random.uniform(-1 * noise_level, noise_level, solution.y.T.shape)
            out_data[:, 1:] = solution.y.T + noise
        else:
            out_data[:, 1:] = solution.y.T

        if save_to_file:
            np.save(out_file, out_data)
        else:
            return out_data


def reshape_data_model1(in_data, out_file=None, save_to_file=False):
    """
    ____________________________
    Build new data by combining data points from in_data.
    The output can be used to train and test NeuralNetworkModel1.
    ____________________________
    """
    num_y = len(in_data[0])-1
    rows = len(in_data)
    out_data = np.empty((rows*(rows-1)//2, 2+2*num_y))  # 2 = len([xi, xi+1])
    n = 0
    for i in range(rows):
        for j in range(i+1, rows):
            out_data[n][0] = in_data[i][0]
            out_data[n][1] = in_data[j][0]
            for k in range(num_y):
                out_data[n][2+k] = in_data[i][1+k]
                out_data[n][2+num_y+k] = in_data[j][1+k]
            n += 1
    if save_to_file:
        np.save(out_file, out_data)
    else:
        return out_data


def reshape_data_model2(in_data, func, out_file=None, save_to_file=False):
    """
    ____________________________
    Build new data by combining data points from in_data.
    The output can be used to train and test NeuralNetworkModel2.
    ____________________________
    """
    num_y = len(in_data[0])-1
    rows = len(in_data)
    out_data = np.empty((rows*(rows-1)//2, 1+3*num_y))
    n = 0
    for i in range(rows):
        for j in range(i+1, rows):
            out_data[n][0] = in_data[j][0] - in_data[i][0]
            out_data[n][1:1+num_y] = in_data[i][1:]
            out_data[n][1+num_y:1+2*num_y] = func(in_data[i][0], in_data[i][1:])
            out_data[n][1+2*num_y:1+3*num_y] = in_data[j][1:]
            n += 1
    if save_to_file:
        np.save(out_file, out_data)
    else:
        return out_data


def create_random_t(t_start, t_end, number_t):
    """
    ____________________________
    Create a set of random t points.
    ____________________________
    """
    t = np.random.uniform(low=t_start, high=t_end, size=number_t)
    t = np.sort(t)
    return t


class _TestODE1(DifferentialEquation):
    @staticmethod
    def func(t, y):
        return -y


class _TestODE2(DifferentialEquation):
    @staticmethod
    def func(t, y):
        return 3/2 * y / (t+1) + np.sqrt(t+1)


class LinearODE1(DifferentialEquation):
    @staticmethod
    def func(t, y):
        return -y/2 + 1/2 * np.exp(t/3)


class LodkaVolterra(DifferentialEquation):
    @staticmethod
    def func(x, y):
        return np.array([y[0] - y[0]*y[1], -y[1] + y[0]*y[1]])


class Kepler(DifferentialEquation):
    @staticmethod
    def func(t, y):
        return np.array([y[2], y[3], -y[0] / (y[0]**2 + y[1]**2)**(3/2), -y[1] / (y[0]**2 + y[1]**2)**(3/2)])


class VanDerPol(DifferentialEquation):
    @staticmethod
    def func(t, y, mu=1):
        return np.array([y[1], mu*(1 - y[0]**2) * y[1] - y[0]])


# Example
def main():
    diff_eq = VanDerPol(t_0=0, t_end=10, y_0=[1, 2])
    t_points = create_random_t(0, 10, number_t=100)
    data = diff_eq.integrate(t_points=t_points, noise_level=0)
    reshaped_data = reshape_data_model2(data, diff_eq.func, save_to_file=False)
    print(reshaped_data)


if __name__ == '__main__':
    main()
