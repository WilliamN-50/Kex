import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class DifferentialEquation:
    def __init__(self, t_0, t_end, y_0):
        self.t_0 = t_0
        self.t_end = t_end
        self.y_0 = y_0

    def func(self, x, y):
        pass

    def integrate(self, method="RK45", t_points=None):
        sol = solve_ivp(self.func, [self.t_0, self.t_end], self.y_0, method=method, t_eval=t_points)
        return sol

    def integrate_and_save(self):
        pass


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
    d_e0 = Diff_eq_2(t_0=0, t_end=10, y_0=[1])
    sol_0 = d_e0.integrate(t_points=np.arange(0, 10, 0.1))
    out_data = np.array([sol_0.t, sol_0.y[0]]).T
    # print(out_data)
    np.savetxt("data_outdata.csv", out_data)



if __name__ == '__main__':
    main()