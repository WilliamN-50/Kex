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


def euler_method(nn_tr_te, device, t0, y0, h0, tol, diff_eq, num_step):
    t = np.array([t0])
    y = np.array([y0])
    y_euler = np.array([y0])
    h = np.array([h0])
    t_step = 0
    for i in range(num_step):
        data = torch.tensor([t[i], t[i]+h[i], h[i], y[i]]).to(device).float()
        nn_e = nn_tr_te.model(data).cpu()
        nn_e = nn_e.detach().numpy()
        tao = nn_e*h[i]**2/2
        # print(h)
        # print([0.9*h[i]*min(max((tol/(2*tao)**(1/2), 0.3)), 2)])
        h = np.append(h, 0.9*h[i]*min(max((tol/(2*tao)**(1/2), 0.3)), 2))
        # print(h)
        t = np.append(t, t[i]+h[i+1])  # t[i+1]
        y = np.append(y, y[i] + h[i+1]*diff_eq.func(t[i], y[i]) + h[i+1]**2 * nn_e)  # y[i+1]
        y_euler = np.append(y_euler, y[i] + 0.1*diff_eq.func(t_step, y[i]))
        t_step += 0.1
    return t, y, y_euler


def main():
    t = np.arange(0, 10, 0.1)
    diff_eq = Diff_eq_2(0, 10, [1])
    data_integrate = diff_eq.integrate(t_points=t)
    data_input = diff_eq.reshape_data(data_integrate)

    batch_size = 500
    device = "cpu"

    nn_tr_te = NN_model.TrainAndTest(diff_eq, data_input, batch_size, device, train_ratio=0.85, lr=1e-3)
    for i in range(10):
        print("____________________")
        print("epoch:{}".format(i + 1))
        print("____________________")
        nn_tr_te.nn_train()
        nn_tr_te.nn_test()

    t_pred, y_pred, y_euler = euler_method(nn_tr_te=nn_tr_te, device=device, t0=0, y0=1, h0=0.5, tol=0.1, diff_eq=diff_eq, num_step=100)

    t2 = np.arange(0, 10, 0.15)
    y = np.zeros(t2.shape)
    y[0] = 1
    for i in range(len(t2) - 1):
        h = t2[i + 1] - t2[i]
        # print(y[i, 0])
        data = torch.tensor([t2[i], t2[i + 1], h, y[i]]).to(device).float()
        nn_e = nn_tr_te.model(data).cpu()
        nn_e = nn_e.detach().numpy()
        y[i + 1] = y[i] + h * diff_eq.func(t2[i], y[i]) + h ** 2 * nn_e

    plt.plot(t_pred, y_pred, label='prediction')
    plt.plot(t, y_euler[:-1], label='Euler')
    plt.plot(t2, y, label="nntest")
    plt.plot(data_integrate[:, 0], data_integrate[:, 1], label='target')
    # plt.plot(data_integrate[:, 0], data_integrate[:, 2], label='target')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
