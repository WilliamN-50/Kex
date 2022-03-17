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
        temp1 = np.array([x_first, x_second, delta_x])
        temp1 = np.append(temp1, temp0)
        data_temp = torch.tensor(temp1).float()
        nn_e = model(data_temp)
        nn_e = nn_e.detach().numpy()
        n.append(list(nn_e))
    r = np.array(r)
    n = np.array(n)
    return r, n

def main():
    diff_eq = Diff_eq_1(0, 10, [1, 2])
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("eq_1_model_50.pth"))
    model.eval()

    t2 = np.arange(0, 10, 0.15)
    y = np.zeros((2, len(t2)))
    y[:, 0] = [1, 2]
    for i in range(len(t2)-1):
        h = t2[i+1] - t2[i]
        # print(y[i, 0])
        data = torch.tensor([t2[i], t2[i+1], h, y[0, i], y[1, i]]).to(device).float()
        nn_e = model(data).cpu()
        nn_e = nn_e.detach().numpy()
        y[:, i+1] = y[:, i] + h*diff_eq.func(t2[i], y[:, i]) + h**2 * nn_e

    t = np.arange(0, 10, 0.1)
    data_integrate = diff_eq.integrate(t_points=t)
    # print(data_integrate)
    r, n = r_n_error(model, data_integrate, diff_eq.func)
    # plt.plot(data_integrate[:-1, 0], r[:, 0], label="R1")
    # plt.plot(data_integrate[:-1, 0], n[:, 0], "--", label="N1")
    plt.plot(data_integrate[:-1, 0], r[:, 1], label="R2")
    plt.plot(data_integrate[:-1, 0], n[:, 1], "--", label="N2")
    plt.legend()
    plt.show()

    plt.plot(t2, y[0, :], label='prediction')
    plt.plot(t2, y[1, :], label='prediction')
    plt.plot(data_integrate[:, 0], data_integrate[:, 1], label='target')
    plt.plot(data_integrate[:, 0], data_integrate[:, 2], label='target')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()