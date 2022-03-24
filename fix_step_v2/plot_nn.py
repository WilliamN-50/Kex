import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import NN_model
import torch


class Diff_eq_0(gd.DifferentialEquation):
    def func(self, t, y):
        return -y


class Diff_eq_1(gd.DifferentialEquation):
    def func(self, x, y):
        return np.array([y[0] - y[0]*y[1], -y[1] + y[0]*y[1]])


class Diff_eq_2(gd.DifferentialEquation):
    def func(self, t, y):
        return 3/2 * y/(t+1) + np.sqrt(t+1)

class Diff_eq_Van_der(gd.DifferentialEquation):
    def func(self, t, y):
        return np.array([y[1], (1-y[0]**2)*y[1]-y[0]])


def r_n_error(model, data, func):
    r = []
    n = []

    for i in range(data.shape[0]-1):
        x_first = data[i, 0]
        x_second = data[i+1, 0]
        delta_x = x_second - x_first
        y_first = data[i, 1:]
        y_second = data[i+1, 1:]
        r_temp = 1/delta_x**2 * (y_second - y_first - delta_x * func(x_first, y_first))
        r.append(list(r_temp))

        temp0 = np.array([y_first[j] for j in range(len(y_first))])
        temp1 = np.array([x_first, x_second])
        temp1 = np.append(temp1, temp0)
        data_temp = torch.tensor(temp1).float()
        nn_e = model(data_temp)
        nn_e = nn_e.detach().numpy()
        n.append(list(nn_e))
    r = np.array(r)
    n = np.array(n)
    return r, n


def main():
    diff_eq = Diff_eq_Van_der(0, 25, [1, 2])
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    # model.load_state_dict(torch.load("../trained_model/eq_van_der_model_Adam_1_2_1000p_noise1.pth"))
    model.load_state_dict(torch.load("eq_van_der_model_Adam_no_noise_1_2_1000p_100ep_lr5_10_4.pth"))
    model.eval()

    t = np.arange(0, 25, 0.1)
    data_integrate = diff_eq.integrate(t_points=t)
    # print(data_integrate)
    r, n = r_n_error(model, data_integrate, diff_eq.func)
    plt.plot(data_integrate[:-1, 0], r[:, 0], label="R of y1")
    plt.plot(data_integrate[:-1, 0], n[:, 0], "--", label="N of y1")
    plt.plot(data_integrate[:-1, 0], r[:, 1], label="R of y2")
    plt.plot(data_integrate[:-1, 0], n[:, 1], "--", label="N of y2")
    plt.title("R and N")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()