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


def euler_method(model, t0, t_end, y0, h, tol, diff_eq, device, fix_step=False):
    t = [t0]
    y = [y0]
    # h = np.array([h0])
    while t[-1] < t_end-h:
        temp0 = np.array([y[-1][j] for j in range(diff_eq.num_y)])
        temp1 = np.array([t[-1], t[-1]+h, h])
        temp1 = np.append(temp1, temp0)

        #
        data = torch.tensor(temp1).to(device).float()
        nn_e = model(data).cpu()
        nn_e = nn_e.detach().numpy()

        #
        if not fix_step:
            h = adaptive_euler_h(h, tol, nn_e)

        #
        t.append(t[-1]+h)  # t[i+1]
        y_temp = np.array(y[-1])
        y_temp = y_temp + h * diff_eq.func(t[-1], y_temp) + h**2 * nn_e  # y[i+1]
        if fix_step:
            y_temp = np.array(y[-1])
            y_temp = y_temp + h * diff_eq.func(t[-1], y_temp)
        y.append(list(y_temp))

    t = np.array(t)
    y = np.array(y)
    return t, y


def adaptive_euler_h(h, tol, nn_e):
    tao = max(abs(nn_e * h ** 2 / 2))
    h = 0.9 * h * min(max(tol / (2 * tao) ** (1 / 2), 0.3), 2)
    return h


def main():
    # t = np.arange(0, 15, 0.13)
    # print(t.shape[0])
    diff_eq = Diff_eq_1(0, 15, [2, 1])
    # t_points = diff_eq.create_t(number_t=1000)
    t_points = np.arange(0, 15, 0.13)
    data_integrate = diff_eq.integrate(t_points=t_points)
    # data_input = diff_eq.reshape_data(data_integrate)

    # batch_size = 500
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("eq_1_model_50.pth"))
    model.eval()

    t_pred, y_pred = euler_method(model=model, t0=0, t_end=15, y0=[2, 1], h=0.5, tol=0.1, diff_eq=diff_eq, device=device)
    t_fix, y_fix_euler = euler_method(model=model, t0=0, t_end=15, y0=[2, 1], h=0.13, tol=0.1, diff_eq=diff_eq, device=device, fix_step=True)
    # print("t", t_fix)
    # print("y", y_pred)
    y_ref_pred = diff_eq.integrate(t_points=t_pred)
    ref_error_nn1 = np.abs(y_pred[:, 0] - y_ref_pred[:, 1])
    ref_error_nn2 = np.abs(y_pred[:, 1] - y_ref_pred[:, 2])

    ref_euler1 = np.abs(y_fix_euler[:, 0] - data_integrate[:, 1])
    ref_euler2 = np.abs(y_fix_euler[:, 1] - data_integrate[:, 2])

    # ref_euler = np.abs(y_fix_euler[:-1] - data_integrate[:, 1])
    plt.plot(t_pred, ref_error_nn1, label="relative error for prediction fun1")
    plt.plot(t_pred, ref_error_nn2, label="relative error for prediction fun2")
    plt.plot(t_fix, ref_euler1, label="relative error for fix euler fun1")
    plt.plot(t_fix, ref_euler2, label="relative error for fix euler fun2")
    # plt.plot(t, ref_euler, label="relative error for euler")
    plt.legend()
    plt.show()

    """ # t_pred with fix step size.
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
    """

    plt.plot(t_pred, y_pred[:, 0], label='prediction func1')
    plt.plot(t_pred, y_pred[:, 1], label='prediction func2')
    # plt.plot(t, y_euler[:-1], label='Euler')
    # plt.plot(t2, y, label="nntest")
    # plt.plot(t_pred, y_ref[:, 1], label='target')
    plt.plot(data_integrate[:, 0], data_integrate[:, 1], label='target')
    plt.plot(data_integrate[:, 0], data_integrate[:, 2], label='target')
    plt.legend()
    plt.show()

    # print(y_pred)
    # print(y_ref)
    # print(y_euler)


if __name__ == '__main__':
    main()