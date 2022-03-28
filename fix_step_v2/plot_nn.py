import matplotlib.pyplot as plt
import numpy as np
import GenerateData as gd
import NN_model
import torch


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


def plot_r_and_n(num_y, t, r, n, title):
    for i in range(num_y):
        plt.plot(t[:-1, 0], r[:, i], label="R of y"+str(i+1))
        plt.plot(t[:-1, 0], n[:, i], "--", label="N of y" + str(i + 1))

    plt.title(title)
    plt.legend()
    plt.show()


def plot_r_and_n_Kepler(t, r, n, title_p, title_q):
    plt.subplot(1, 2, 1)
    for i in range(2):
        plt.plot(t[:-1, 0], r[:, i], label="R of p" + str(i + 1))
        plt.plot(t[:-1, 0], n[:, i], "--", label="N of p" + str(i + 1))
    plt.title(title_p)
    plt.xlabel("t values")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(2, 4):
        plt.plot(t[:-1, 0], r[:, i], label="R of q" + str(i-1))
        plt.plot(t[:-1, 0], n[:, i], "--", label="N of q" + str(i-1))

    plt.title(title_q)
    plt.xlabel("t values")
    plt.ylabel("Error")

    plt.legend()
    plt.show()


def main():
    # diff_eq = gd.Kepler(t_0=0, t_end=25, y_0=[0.5, 0, 0, np.sqrt(3)])
    diff_eq = gd.VanDerPol(t_0=0, t_end=25, y_0=[1, 2])
    device = "cpu"
    model = NN_model.NeuralNetwork(diff_eq.num_y)
    model.load_state_dict(torch.load("../trained_model/vanderpol_50_lr_5e-4_no_noise.pth"))
    # model.load_state_dict(torch.load("VanderPol_5e4_1000p_30ep.pth"))
    # model.load_state_dict(torch.load("eq_van_der_model_Adam_no_noise_1_2_1000p_100ep_lr5_10_4.pth"))
    model.eval()

    t = np.arange(0, 25, 0.1)
    data_integrate = diff_eq.integrate(t_points=t)
    # print(data_integrate)
    r, n = r_n_error(model, data_integrate, diff_eq.func)
    plot_r_and_n(diff_eq.num_y, data_integrate, r, n, "R and N")
    # plot_r_and_n_Kepler(data_integrate, r, n, "R and N for p", " R and N for q")


if __name__ == '__main__':
    main()
