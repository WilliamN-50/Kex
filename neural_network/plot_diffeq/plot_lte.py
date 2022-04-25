import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_network.model import diffeqnetworkmodel1 as model1, diffeqnetworkmodel2 as model2
from neural_network.model import differentialequations as deq


def model_lte(model, in_data, func):
    """
    ____________________________
    Calculates the local truncation error of the NeuralNetwork model and Euler forward.
    ____________________________
    """
    num_y = len(in_data[0])-1
    nn_lte = np.empty((len(in_data)-1, num_y))
    lte = np.empty(nn_lte.shape)

    for i in range(len(in_data)-1):
        if type(model) == model1.NeuralNetworkModel1:
            data = deq.reshape_data_model1(in_data[i:i+2])
            torch_data = torch.tensor(data[0, :2+num_y]).float()
            lte[i, :] = model1.euler_local_truncation_error(data[0], func, num_y)
        else:
            data = deq.reshape_data_model2(in_data[i:i+2], func)
            torch_data = torch.tensor(data[0, :1+2*num_y]).float()
            lte[i, :] = model2.euler_local_truncation_error(data[0], num_y)
        nn_e = model(torch_data)
        nn_lte[i, :] = nn_e.detach().numpy()
    return nn_lte, lte


def exact_lte(in_data, func):
    """
    ____________________________
    Calculates the local truncation error of Euler forward, implicit Euler, and Euler-Cromer.
    ____________________________
    """
    lte_forward_euler = np.empty((len(in_data)-1, len(in_data[0])-1))
    lte_implicit_euler = np.empty(lte_forward_euler.shape)

    for i in range(len(in_data)-1):
        t_first = in_data[i, 0]
        t_second = in_data[i+1, 0]
        h = t_second - t_first
        y_first = in_data[i, 1:]
        y_second = in_data[i+1, 1:]

        lte_fe = 1 / h**2 * (y_second - y_first - h * func(t_first, y_first))
        lte_ie = 1 / h**2 * (y_second - y_first - h * func(t_second, y_second))

        lte_forward_euler[i, :] = lte_fe
        lte_implicit_euler[i, :] = lte_ie

    return lte_forward_euler, lte_implicit_euler


def plot_lte(t, lte, num_y, label, marker):
    for i in range(num_y):
        plt.plot(t[:-1], lte[:, i], marker, label=f"{label} of y{i+1}")


def plot_multi_lte(t, lte_list, num_y, labels, markers):
    """
    ____________________________
    Plots the local truncation errors in lte_list.
    ____________________________
    """
    for i in range(len(lte_list)):
        plot_lte(t, lte_list[i], num_y, labels[i], markers[i])


def plot_lte_hamiltonian(t, lte, nn_lte, num_y, title_p, title_q):
    """
    ____________________________
    Plots the local truncation errors for a hamiltonian system.
    ____________________________
    """
    plt.subplot(1, 2, 1)
    for i in range(num_y//2):
        plt.plot(t[:-1], lte[:, i], label="lte of q" + str(i+1))
        plt.plot(t[:-1], nn_lte[:, i], "--", label="nn_lte of q" + str(i+1))
    plt.title(title_q)
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(num_y//2, num_y):
        plt.plot(t[:-1], lte[:, i], label="lte of p" + str(i+1-num_y//2))
        plt.plot(t[:-1], nn_lte[:, i], "--", label="nn_lte of p" + str(i+1-num_y//2))
    plt.title(title_p)
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.legend()


def main():
    # Properties of differential equation
    t_0 = 0
    t_end = 30
    # y_0 = [1]
    # diff_eq = deq.LinearODE1(t_0, t_end, y_0)
    # y_0 = [1, 2]
    # diff_eq = deq.VanDerPol(t_0, t_end, y_0)
    y_0 = [0.5, 0, 0, np.sqrt(3)]
    diff_eq = deq.Kepler(t_0, t_end, y_0)

    # Load model
    filename = "../../trained_model/model2/model2_Kepler_1000p_500batch_75ep_lr5e-4.pth"
    # filename = "model1_Kepler_1000p_500batch_75ep_lr5e-4.pth"
    # filename = "../trained_model/Kepler_no_noise_0_10_1000p_100ep_1e3.pth"
    model = model2.NeuralNetworkModel2(diff_eq.num_y)
    model.load_state_dict(torch.load(filename))
    model.eval()

    # Construct data
    h = 0.1
    t = np.arange(t_0, t_end, h)
    y = diff_eq.integrate(t_points=t)

    # Exact lte
    # lte_fe, lte_ie = exact_lte(y, diff_eq.func)
    # lte_exact = [lte_fe, lte_ie]
    # label_exact = ["Euler Forward lte", "Implicit Euler lte"]
    # markers_exact = ["r", "b"]

    # Model lte
    nn_lte, lte = model_lte(model, y, diff_eq.func)
    label_model = ["reference LTE", "model LTE"]
    markers_model = ["-", "--"]
    lte_model = [lte, nn_lte]

    # title = "LTE of Linear ODE, model 1"

    # plot_multi_lte(t, lte_model, diff_eq.num_y, label_model, markers_model)
    # plot_multi_lte(t, lte_exact, diff_eq.num_y, label_exact, markers_exact)
    plot_lte_hamiltonian(t, lte, nn_lte, diff_eq.num_y, "LTE of p, h=0.1, model 1", "LTE of q, h=0.1, model 1")

    # plt.title("LTE of Van der Pol, model 1")
    plt.xlabel("t")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()