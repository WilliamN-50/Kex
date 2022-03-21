import numpy as np
from scipy.integrate import solve_ivp


class DifferentialEquation:
    """
    ____________________________
    The DifferentialEquation class.
    Generates data for the NeuralNetwork model.
    ____________________________
    """
    def __init__(self, t_0, t_end, y_0):
        self.t_0 = t_0
        self.t_end = t_end
        self.y_0 = y_0
        self.num_y = len(y_0)

    def func(self, x, y):
        pass

    def create_t(self, number_t):
        """
        ____________________________
        Create the set of the t points.
        ____________________________
        """
        t = []
        for i in range(number_t):
            t.append(15*np.random.random())

        t.sort()
        return t

    def integrate(self, method="RK45", rtol=10**(-6), t_points=None, noise_level=0.0, out_file=None, save_to_file=False):
        """
        ____________________________
        Integrates the differential equation.
        ____________________________
        """
        solution = solve_ivp(self.func, [self.t_0, self.t_end], self.y_0, method=method, t_eval=t_points, rtol=rtol)
        rows = solution.t.shape[0]
        noise = np.random.uniform(-1*noise_level, noise_level, solution.y.T.shape)
        out_data = np.zeros((rows, 1+self.num_y))
        out_data[:, 0] = solution.t.T
        out_data[:, 1:] = solution.y.T + noise

        if save_to_file:
            np.save(out_file, out_data)
        else:
            return out_data

    def reshape_data(self, in_data, out_file=None, save_to_file=False):
        """
        ____________________________
        Build a new data structure by combining data points from in_file.
        The output will be used to train and test the NeuralNetwork model.
        ____________________________
        """
        rows = in_data.shape[0]  # indata antalet [x,y]
        out_data = np.zeros((rows*(rows-1)//2, 3+2*self.num_y))  # 3 = the number of [x0, x1, delta_X]
        n = 0
        for i in range(rows):
            for j in range(i+1, rows):
                out_data[n][0] = in_data[i][0]
                out_data[n][1] = in_data[j][0]
                out_data[n][2] = in_data[j][0] - in_data[i][0]
                for k in range(self.num_y):
                    out_data[n][3+k] = in_data[i][1+k]
                    out_data[n][3+self.num_y+k] = in_data[j][1+k]
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
        return np.array([y[0] - y[0]*y[1], -y[1] + y[0]*y[1]])


class Diff_eq_2(DifferentialEquation):
    def func(self, t, y):
        return 3/2 * y/(t+1) + np.sqrt(t+1)


def main():
    d_e0 = Diff_eq_1(t_0=0, t_end=15, y_0=[2, 1])
    t_points = d_e0.create_t(number_t=1000)
    data = d_e0.integrate(t_points=t_points, noise_level=0.1)
    reshaped_data = d_e0.reshape_data(data, out_file='outfile_exempel_noise.npy', save_to_file=True)

    print(reshaped_data)


if __name__ == '__main__':
    main()